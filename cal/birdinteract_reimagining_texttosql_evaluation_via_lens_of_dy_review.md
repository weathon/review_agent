=== CALIBRATION EXAMPLE 85 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
- **Does the title accurately reflect the contribution?** Yes. "Dynamic Interactions" directly captures the core shift from static transcripts to live, budget-constrained, multi-turn evaluation.
- **Does the abstract clearly state the problem, method, and key results?** Yes. It cleanly identifies the limitations of prior work (static context, SELECT-only), outlines the three pillars of the benchmark (environment, two settings, CRUD task suite), and reports concrete empirical findings (low success rates for frontier models, ITS, memory grafting insights).
- **Are any claims in the abstract unsupported by the paper?** No. All claims are substantiated in Sections 4 and 5. The mention of "GPT-5" aligns with the experimental setup, though readers should be aware that performance metrics for unreleased or pre-release models depend on specific API routing/timing.

### Introduction & Motivation
- **Is the problem well-motivated? Is the gap in prior work clearly identified?** Strongly motivated. The critique of existing multi-turn benchmarks for relying on static transcripts (Section 1) and ignoring DDM/DDL operations is precise and addresses a recognized pain point in the NLIDB community.
- **Are the contributions clearly stated and accurate?** Yes. The three contributions map directly to the technical sections: the environment/simulator (Sec 3), the evaluation settings (Sec 4), and the annotated dataset (Sec 3.2-3.4).
- **Does the introduction over-claim or under-sell?** Appropriately calibrated. It positions the work as an evaluation benchmark rather than a new generation architecture, which is standard and acceptable for ICLR.

### Method / Approach
- **Is the method clearly described and reproducible?** Generally yes. The formalization in Equation (1) is standard. The annotation pipeline (Sec 3.2), two-stage simulator (Sec 3.3), and budget constraints (Sec 4.1, 4.2) are detailed. Reproducibility commitments (fresh Docker instances, prompts in App R, action costs in Table 9) are strong.
- **Are key assumptions stated and justified?** The assumption of exactly two sub-tasks per task (`n=2`, Section 2) is a simplifying design choice for tractability, but the authors should clarify why longer chains (n>2) were excluded, as real-world tasks often involve deeper iterative refinement. Additionally, the reward weighting in Appendix F.2 (0.7/0.5 for sub-task 1, 0.3/0.2 for sub-task 2) is stated but lacks ablation to justify why this specific distribution captures "priority" better than alternatives (e.g., 0.6/0.4 or unweighted).
- **Are there logical gaps in the derivation or reasoning?** The budget formula for `a`-Interact (`B = B_base + 2m_amb + 2λ_pat`) is clear, but the justification for the `2x` multiplier compared to `c`-Interact is not explicitly derived. A brief rationale for why agentic exploration requires precisely double the ambiguity budget would strengthen the design justification.
- **Are there edge cases or failure modes not discussed?** The two-stage simulator (Section 3.3) maps questions to `AMB`, `LOC`, or `UNA`. If a model asks a highly creative but valid clarification that falls outside the `LOC` AST retrieval scope and isn't pre-annotated, the simulator may incorrectly reject it or provide a suboptimal response. The robustness of the AST-based `LOC()` function to out-of-distribution phrasing is not empirically tested.

### Experiments & Results
- **Do the experiments actually test the paper's claims?** Yes. The low success rates validate the difficulty of dynamic interaction. The `c` vs. `a` mode comparison, memory grafting, and ITS experiments directly probe the claims about communication effectiveness and interaction scaling.
- **Are baselines appropriate and fairly compared?** For an ICLR benchmark paper, baselines should include not only frontier LLMs under ReAct/vanilla prompting, but also state-of-the-art *specialized multi-turn text-to-SQL agents* (e.g., adaptations of MAC-SQL, DTS-SQL, or agent frameworks explicitly trained for clarifications). Using only generic prompting may understate what purpose-built systems can achieve, potentially inflating the perceived "impossibility" of the benchmark.
- **Are there missing ablations that would materially change conclusions?** Yes. The action cost scheme (Table 9: `ask`=2, `submit`=3, others ≤1) heavily influences agent behavior in `a`-Interact. An ablation varying these costs (e.g., uniform cost vs. current graduated cost) would clarify whether the observed "trial-and-error vs. exploration" behavioral patterns (Section 5.2, Figure 11-13) are inherent to model capabilities or artifacts of the specific cost penalties.
- **Are error bars / statistical significance reported?** No. Section 5 explicitly states "single runs due to cost." While deterministic decoding (`temperature=0`) is used, API latency, routing variability, and prompt parsing non-determinism introduce variance. ICLR standards for benchmarks typically require at least 3 runs or bootstrapped confidence intervals, especially for success rate metrics where variance can be high. The lack of error bars is a notable weakness.
- **Do the results support the claims made, or are they cherry-picked?** Results appear consistent across the LITE and FULL sets (Table 2, Table 10). The BI vs. DM analysis and action distribution insights are well-supported. No evidence of cherry-picking.
- **Are datasets and evaluation metrics appropriate?** Yes. Soft exact-match for BI and case-specific execution scripts for DM align with BIRD/LIVESQLBENCH standards. The Normalized Reward metric is useful for partial credit.

### Writing & Clarity
- **Are there sections that are confusing or poorly explained?** The budget formulas in Sections 4.1 and 4.2 are clear, but the transition between theoretical budget constraints and practical implementation costs could be tighter. Figure 1 contains significant parsing artifacts in the text extraction, but the visual flow of interaction is conceptually clear from the description.
- **Are figures and tables clear and informative?** Table 2 and Table 3 are highly informative. The action distribution heatmaps (Figures 11-12) and turn-by-turn trajectories (Figure 13) effectively communicate behavioral differences. References to these figures are accurate.

### Limitations & Broader Impact
- **Do the authors acknowledge the key limitations?** Appendix A notes the focus on text-to-SQL and generalizability. The high cost of multi-turn API calls is implicitly acknowledged via single-run evaluation. However, explicit discussion of the simulator's limitations as an evaluation proxy is missing.
- **Are there fundamental limitations they missed?** 
  1. **Simulator Rigidity:** Even with the two-stage guard, the simulator cannot model genuine user behaviors like mid-conversation goal shifts, contradictory feedback, or providing incorrect domain knowledge when pressed. This limits ecological validity for truly open-ended `a`-Interact scenarios.
  2. **Fixed Horizon:** Restricting tasks to exactly two sub-tasks (`n=2`) simplifies evaluation but fails to stress-test long-horizon state management and context window degradation over 5-10+ meaningful turns, which is common in production DB interactions.
- **Are there failure modes or negative societal impacts not discussed?** The paper does not discuss the risk of *simulator bias* penalizing novel but valid interaction strategies. If the benchmark becomes a standard, models may overfit to the specific `AMB/LOC/UNA` taxonomy and AST structures rather than learning robust, generalizable communication. This alignment risk warrants mention.

### Overall Assessment
This paper presents a timely and technically rigorous benchmark that addresses a genuine gap in text-to-SQL evaluation: the lack of dynamic, multi-turn, CRUD-capable interactive assessment. The two-stage function-driven user simulator is a clever and well-validated engineering contribution that mitigates ground-truth leakage more effectively than standard baselines. The empirical findings, particularly the Interaction Test-Time Scaling trend and the "memory grafting" insight, offer valuable direction for the community. However, the evaluation has notable methodological gaps by ICLR standards. The reliance on single-run experiments without error bars or statistical significance testing weakens the robustness of the reported metrics. Additionally, the baselines consist only of frontier LLMs under standard prompting/ReAct, omitting specialized multi-turn SQL agents that could establish a stronger performance ceiling. Finally, the heuristic nature of the action costs and reward weightings lacks ablation to prove that the observed behavioral patterns are intrinsic to the models rather than artifacts of the cost scheme. Despite these concerns, the benchmark construction is thorough, the simulator design is novel, and the dataset will be valuable for the community. I recommend acceptance provided the authors address the statistical rigor (multiple runs/error bars), broaden their agent baselines or clarify why generic agents suffice, and include an ablation of the cost/reward parameters to solidify their behavioral conclusions.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces BIRD-INTERACT, a benchmark designed to evaluate LLM-based text-to-SQL systems under dynamic, multi-turn interactions rather than static conversation transcripts. The framework couples a function-driven user simulator (to prevent ground-truth leakage and ensure controllability) with two evaluation paradigms: a protocol-guided conversational setting (`c`-Interact) and a budget-constrained agentic setting (`a`-Interact). Empirical analysis across 900 tasks demonstrates that even frontier models struggle with ambiguity resolution, state-dependent follow-ups, and efficient action sequencing, highlighting strategic interaction as a critical bottleneck for database reasoning.

### Strengths
1. **Addresses a Meaningful Evaluation Gap:** The shift from static multi-turn transcripts to dynamic, simulator-driven interactions directly tackles a well-known limitation in text-to-SQL research. By requiring models to actively resolve ambiguities, handle execution errors, and adapt to evolving goals, the benchmark better reflects production database assistant workflows (Section 1, Figure 1).
2. **Methodologically Sound Simulator Design:** The two-stage function-driven user simulator (Sec 3.3, Table 11, Figure 6) is a strong technical contribution. By first classifying clarification intent into constrained symbolic actions (AMB, LOC, UNA) before generating responses, the authors effectively mitigate ground-truth leakage and improve simulator alignment with human users (0.84 Pearson correlation vs. 0.61 baseline).
3. **Actionable Empirical Insights:** The paper goes beyond reporting success rates. The "Memory Grafting" experiment (Fig 5) convincingly isolates communication failure from SQL generation capability, while the Interaction Test-Time Scaling (ITS) analysis (Sec 5.2, Fig 4) and action distribution patterns (Figs 11-13) provide practical design guidelines for future agentic systems (e.g., balancing exploration vs. exploitation under budget constraints).
4. **Strong Reproducibility Infrastructure:** The authors provide deterministic decoding settings, explicit budget formulas, Docker-based isolated PostgreSQL environments, and complete prompt templates (Appendix R). The commitment to releasing code, trajectories, and test scripts under a permissive license aligns well with ICLR's reproducibility standards.

### Weaknesses
1. **Limited Statistical Robustness:** Due to computational cost, all experiments are run once (Appendix I.3). Given known non-determinism in LLM API responses even at `temperature=0`, reporting single-point estimates without variance or confidence intervals limits the reliability of comparative claims. ICLR typically expects either multiple runs or explicit justification via lower-cost proxy models.
2. **Under-Justified Heuristics in Evaluation Design:** Several key design choices lack rigorous sensitivity analysis or ablation. The normalized reward weighting (70% priority / 30% follow-up, App F.2), the budget formula (`B = B_base + 2m_amb + 2λ_pat`, Sec 4.2), and discrete action costs (Table 9) appear heuristic. Without parameter sweeps or justification, it's unclear if conclusions hold under alternative budget/cost regimes.
3. **Baseline Coverage:** The evaluation exclusively tests prompt-engineered frontier LLMs in ReACT or conversational modes. It lacks comparisons with specialized agentic text-to-SQL frameworks (e.g., MAC-SQL, DAIL-SQL, or reasoning-optimized chains adapted for multi-turn). This makes it difficult to attribute low performance to the task difficulty versus suboptimal prompting strategies.
4. **Transparency on "GPT-5" Designation:** The paper reports results for "GPT-5" (Table 2, App I.2), which is not a publicly released or verifiable model at the time of review. ICLR reviewers prioritize reproducibility and verifiable baselines; using an unreleased or internal model alias undermines the ability of the community to replicate claims.

### Novelty & Significance
**Novelty:** High. The benchmark successfully bridges dynamic interaction, full-spectrum CRUD operations, and budget-constrained agent evaluation in a unified framework. The two-stage function-driven simulator is a novel and effective safeguard against evaluation leakage, a problem that has plagued prior interactive benchmarks.
**Clarity:** High. The paper is well-structured, with clear problem formulation, thorough annotation taxonomies, and extensively documented appendices. Tables and figures effectively communicate task design, results, and agent behaviors.
**Reproducibility:** High, with minor caveats. The explicit environment setup, prompt release, and deterministic settings are strong. The use of an unreleased model alias and single-run evaluations slightly detract from full reproducibility.
**Significance:** High for the ML/agent community. By demonstrating that strategic interaction and budget management—not just raw generation capability—are critical for database reasoning, the paper identifies a clear research direction and provides a robust testbed for future interactive agent development.

### Suggestions for Improvement
1. **Strengthen Statistical Validity:** Run experiments across ≥3 seeds or use a more affordable open-source proxy (e.g., Qwen-Coder, DeepSeek) to report mean/std or confidence intervals for key metrics (SR, Reward, action costs). If budget constraints persist, explicitly report variance estimates or bootstrapped intervals.
2. **Ablate Heuristic Parameters:** Provide a sensitivity analysis for the budget formula, action costs, and the 70/30 reward split. Show whether model rankings or key conclusions (e.g., ITS trends, action distribution patterns) remain stable under alternative weighting schemes.
3. **Include Agentic Baselines:** Adapt at least one specialized text-to-SQL agent framework (e.g., MAC-SQL, DIN-SQL, or a chain-of-thought schema-linking pipeline) to the `a`-Interact action space. This will contextualize whether low performance stems from interaction complexity or the limitations of zero-shot/few-shot ReACT prompting.
4. **Clarify Model Aliases:** Replace "GPT-5" with a publicly accessible, versioned model alias (e.g., `gpt-4o-2024-05-13`, `o1`, or `claude-3-5-sonnet`) to ensure strict reproducibility. If results are preliminary/internal, clearly mark them as such and focus primary claims on reproducible baselines.
5. **Discuss Error Propagation:** Briefly analyze how partial success or early failures in sub-task 1 impact the evaluation of sub-task 2. Clarify whether the benchmark penalizes cascading errors, retries, or state-correction attempts, as this affects how "state dependency" is practically measured.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-4 only)
1. Benchmark specialized text-to-SQL agents (e.g., MAC-SQL, DAIL-SQL, or Reflexion-based variants) in the _a_-Interact setting, because testing only generic LLMs with basic ReAct prompts conflates inadequate agent scaffolding with true benchmark difficulty.
2. Evaluate models on static, ambiguity-free versions of the same tasks and report the performance delta, otherwise the central claim that dynamic interaction is the primary bottleneck cannot be disentangled from inherent SQL complexity.
3. Run an unconstrained (no budget/turn caps) pilot experiment, because attributing low success rates to model weakness is unconvincing when artificial resource scarcity may be the actual failure trigger.
4. Test the function-driven simulator against adversarial or highly unstructured model queries outside the predefined AMB/LOC/UNA categories, because claims of robust ground-truth leakage prevention are incomplete without out-of-distribution stress testing.

### Deeper Analysis Needed (top 3-4 only)
1. Provide a phase-level error breakdown distinguishing communication failures (misreading simulator replies), state-tracking failures (losing DB context across turns), and SQL generation failures, because lumping all errors under "incomplete ambiguity resolution" obscures whether models truly lack interaction skills or just fail at parsing feedback.
2. Quantify state-dependency impact by directly comparing follow-up success on tasks that modify the DB schema/state versus identical tasks that do not, as the claim that evolving database states add unique difficulty lacks isolated empirical support.
3. Analyze trajectories where offline normalized reward significantly diverges from online success rate to prove that the 70/30 reward weighting accurately prioritizes critical objectives rather than artificially inflating or suppressing model rankings.
4. Correlate clarification question quality with subsequent SQL accuracy to demonstrate that successful models actually ask better questions, rather than succeeding merely by chance or through brute-force SQL guessing within relaxed budgets.

### Visualizations & Case Studies
1. Plot aggregated action-sequence heatmaps across interaction turns for both successful and failed tasks, revealing whether failing models systematically stall during exploration, submit prematurely, or waste budget on redundant queries.
2. Provide side-by-side interaction logs for state-dependent follow-ups, explicitly tracking whether models retain newly created DB artifacts (tables/views/functions) or hallucinate back to the original schema, visually confirming the claimed state-tracking challenge.
3. Show direct human vs. baseline vs. function-driven simulator response comparisons for identical clarification prompts to prove the two-stage design preserves conversational naturalness while enforcing controllability.

### Obvious Next Steps
1. Implement and release a configurable "free-mode" evaluation pipeline alongside the budget-constrained version, because ICLR expects the authors to contextualize constrained results against unconstrained baselines rather than deferring this entirely to future work.
2. Conduct a prompt/ICL sensitivity analysis across all evaluated models, as relying on single generic prompts risks underestimating LLM capability and unfairly attributing prompt-engineering failures to fundamental model limitations.
3. Release the full annotation guidelines, SQL-to-ambiguity mapping rules, and simulator prompt templates alongside the benchmark, because without them, independent researchers cannot verify whether ambiguity injection and simulator responses are reproducible or arbitrarily defined.

# Final Consolidated Review
## Summary
BIRD-INTERACT introduces a dynamic, multi-turn evaluation benchmark for text-to-SQL systems, moving beyond static conversation transcripts to a live, budget-constrained environment. It features a two-stage function-driven user simulator, dual evaluation paradigms (protocol-guided vs. agentic), and 900 tasks spanning full CRUD operations with deliberate ambiguity injection and state-dependent follow-ups. Empirical results reveal that even frontier LLMs struggle severely with strategic interaction and communication, highlighting a critical bottleneck in current database reasoning pipelines.

## Strengths
- **Robust, leakage-resistant simulator design:** The two-stage function-driven user simulator (classifying queries into AMB/LOC/UNA before generation) effectively mitigates ground-truth leakage, a pervasive flaw in prior interactive benchmarks. This is empirically validated via the UserSim-Guard dataset (reducing unanswerable query failure rates from ~67% to <3%) and demonstrates strong human alignment (0.84 Pearson correlation vs. 0.61 for baselines).
- **Actionable, diagnostic empirical findings:** The paper successfully isolates the core bottleneck in interactive SQL generation. The "Memory Grafting" experiment convincingly shows that models like GPT-5 possess strong SQL generation capabilities but fail due to poor interactive communication. Combined with Interaction Test-Time Scaling (ITS) and action distribution analyses, the benchmark provides clear, actionable signals for future agent design.

## Weaknesses
- **Lack of statistical rigor for an evaluation benchmark:** All experimental results are reported as single-run point estimates (Appendix I.3). Even with `temperature=0`, multi-turn API evaluations suffer from parsing variance, routing stochasticity, and context-window stochasticity that accumulate across turns. The absence of multiple seeds, bootstrapped confidence intervals, or variance reporting undermines the reliability of comparative claims and model rankings.
- **Incomplete baseline coverage conflates task hardness with prompt engineering:** The evaluation exclusively tests frontier LLMs under zero/few-shot ReAct or conversational prompting. It omits specialized multi-turn text-to-SQL agents (e.g., fine-tuned pipelines, schema-linking agents explicitly trained on clarification, or dedicated agentic frameworks). Without these, it is unclear whether low success rates reflect fundamental model limitations or simply inadequate scaffolding and prompt design.
- **Heuristic evaluation parameters lack sensitivity analysis:** Key design choices—including the 70/30 priority reward split, the budget formula (`B = B_base + 2m_amb + 2λ_pat`), and the discrete action cost scheme—are presented without justification or ablation. Consequently, behavioral conclusions (e.g., models' preference for trial-and-error over exploration) may be artifacts of the specific cost penalties rather than intrinsic model capabilities.
- **Use of an unverifiable model alias damages reproducibility:** Primary results feature "GPT-5" (Table 2), which is not a publicly released or independently verifiable model. For a benchmark paper positioning itself on reproducibility and open science, relying on an internal or pre-release model alias prevents independent verification of the reported upper bounds.

## Nice-to-Haves
- A fine-grained error taxonomy distinguishing communication failures (misreading simulator feedback), state-tracking degradation (losing context over turns), and raw SQL syntax/logic errors, rather than aggregating them under "incomplete ambiguity resolution."
- An unconstrained ("free-mode") pilot experiment to disentangle model capability limits from artificial budget caps, providing context for the stress-test results.
- Quantitative correlation analysis between the specificity/quality of clarification questions and subsequent SQL success rates, to validate whether models that "ask better" truly perform better.

## Novel Insights
The paper demonstrates that the primary failure mode in production-grade text-to-SQL is not SQL generation capacity, but strategic interaction design. Through memory grafting, the authors show that injecting high-quality clarification histories into weak communicators instantly boosts their performance, proving that the SQL backbone is already competent if properly guided. Furthermore, the action distribution analysis reveals a systemic bias toward expensive, brute-force trial-and-error (`submit`/`ask`) over low-cost environmental probing, indicating that current LLM pre-training priors penalize systematic resource exploration. The monotonic performance gains observed under Interaction Test-Time Scaling further prove that successful database reasoning is less about static capability and more about efficient information-gain strategies under finite computational user patience.

## Potentially Missed Related Work
- Recent specialized agentic SQL frameworks that explicitly train or prompt for schema grounding and clarification handling (e.g., refined versions of MAC-SQL, DTS-SQL, or reasoning-SQL pipelines adapted to ReAct) should be considered as baselines to establish a stronger performance ceiling and contextualize the benchmark's difficulty.
- Literature on budget-aware reasoning control (e.g., dynamic token/compute allocation in multi-hop QA) could strengthen the theoretical grounding of the `a`-Interact cost scheme.

## Suggestions
1. **Strengthen statistical validity:** Run evaluations across ≥3 random seeds or report bootstrapped confidence intervals around Success Rate and Reward metrics. If API costs are prohibitive, conduct multi-seed runs on open-source proxies (e.g., Qwen-Coder, DeepSeek variants) to characterize variance.
2. **Clarify or replace the GPT-5 alias:** Either map it to a publicly accessible, versioned model endpoint, or clearly mark it as preliminary/internal and shift primary empirical claims to fully reproducible baselines.
3. **Incorporate a specialized agentic baseline:** Adapt at least one state-of-the-art, multi-turn text-to-SQL agent framework to the `a`-Interact action space. This is essential to prove the benchmark tests genuine interaction complexity rather than prompt engineering gaps.
4. **Ablate heuristic parameters:** Provide a sensitivity analysis for the reward weighting, budget multipliers, and action costs. Demonstrate that key findings (e.g., action preference shifts, model rankings, ITS trends) remain stable under alternative cost regimes.

# Actual Human Scores
Individual reviewer scores: [6.0, 8.0, 8.0, 8.0]
Average score: 7.5
Binary outcome: Accept
