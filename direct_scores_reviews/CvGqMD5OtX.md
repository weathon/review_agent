## Summary
CHASE-SQL is a multi-agent Text-to-SQL framework that combines three diverse candidate generation strategies—Divide-and-Conquer CoT, Query Plan CoT, and Online Synthetic Example Generation (OS ICL)—with a fine-tuned pairwise binary selection agent to identify the best SQL query from a candidate pool. The system achieves 73.0% execution accuracy on the BIRD benchmark test set, outperforming the previous best published method by ~5.8% (dev) and all undisclosed leaderboard entries, and generalizes to Spider (87.6%) without any target-domain retraining.

---

## Strengths

- **Large, well-validated SOTA margin on BIRD.** The 73.01% dev / 73.0% test results exceed the next-best published method (Distillery+GPT-4o, 67.21% dev / 71.83% test) by nearly 6 points, and top all undisclosed leaderboard entries. The performance on Spider (87.6%) without any Spider-specific training or prompt tuning is further evidence of robust generalization.

- **Query Plan CoT is a creative, well-motivated reasoning strategy.** Translating the database engine's EXPLAIN output into a human-readable format and using it as a reasoning scaffold directly exploits the structure of the task in a way no prior Text-to-SQL CoT method has done. Appendix Fig. 21 provides a concrete case where this method uniquely succeeds where others fail.

- **Online Synthetic Example Generation (OS ICL) is a genuinely novel ICL contribution.** Rather than retrieving from a fixed pool of demonstrations, the system synthesizes many-shot examples conditioned on the target schema and SQL feature distribution *at inference time*. The approach achieves 68.02% single-candidate accuracy with Gemini 1.5 Pro—the best of the three generators—and is shown to be complementary to the CoT methods via the Venn diagram in Fig. 3a.

- **The pairwise selection agent is rigorously validated.** Table 6 documents a consistent ~6% gain over self-consistency across all three generators and two temperatures, and Table 7 shows the ranker-agent alternative underperforms by 7.5%, directly supporting the design choice of pairwise comparisons. The selection agent's robustness to temperature variation (while self-consistency degrades) is an insightful finding about the interaction between diversity and selection quality.

- **Open-source reproducibility path.** Using Mistral Large + fine-tuned Qwen-2.5-coder, the framework reaches 70.33% on BIRD dev—exceeding all prior published work—providing a meaningful community contribution independent of expensive frontier model access.

- **Generator complementarity is empirically demonstrated.** Fig. 3a's Venn diagram concretely shows that each generator solves questions the other two cannot (35, 38, and 38 exclusive successes respectively), justifying the complexity of maintaining all three pipelines.

---

## Weaknesses

- **Algorithm 1 contains a factual inconsistency.** The paper explicitly states (line 66) that the Divide-and-Conquer strategy generates output "using a **single LLM call**," yet Algorithm 1 shows a sequential decomposition (one decomposition call, one call per sub-question in a loop, one assembly call). This is not a single call. For a core algorithmic contribution, this misstatement needs correction and is not a matter of interpretation.

- **Query Plan CoT test-time mechanism is ambiguous.** The paper describes converting EXPLAIN output into human-readable reasoning steps, but never clearly states whether (a) the LLM is prompted to *generate* a query-plan-style rationale from scratch given the question and schema (purely synthetic reasoning), or (b) an actual SQL query is first generated and run through EXPLAIN to obtain a real execution plan used as chain-of-thought context. If interpretation (b), there is a bootstrapping problem—you need a SQL query to get an EXPLAIN plan before generating the SQL. The appendix prompts and figures are cited but inaccessible to reviewers without the appendix. This ambiguity matters because the mechanism underpins one of the paper's three novel contributions.

- **Selection agent training uses GT hints at training time but not at inference—a train-test distribution mismatch.** Section 2.5 states: "for instances where no correct candidate exists, we include the ground truth SQL query in the prompt as a hint to guide the model in generating correct candidates." At inference, no ground truth is available. The reported 71.01% binary accuracy (Table 5) is measured on pairs generated using this protocol, meaning the model was trained on data partially generated with oracle guidance that is unavailable at test time. The impact of this mismatch on generalization is unaddressed.

- **Correctness of OS ICL synthetic examples is not analyzed.** The synthetic SQL examples injected as few-shot demonstrations are themselves LLM-generated and not validated for correctness. Incorrect examples used as demonstrations could systematically mislead generation. The paper neither measures the error rate of the generated examples nor analyzes whether incorrect examples harm downstream SQL generation quality.

- **Total inference cost is uncharacterized.** Each CHASE-SQL query involves: (a) multi-step OS ICL generation (two synthesis passes), (b) 7 candidates × 3 generators = 21 candidates with multi-step DC CoT, (c) up to β=3 fix iterations per candidate, and (d) Algorithm 3's pairwise comparisons (which doubles all pairs for order-bias mitigation). The total LLM call count per query is large, yet no latency, token count, or API cost estimate is provided. This makes the framework's practical deployability opaque and omits a key trade-off dimension for assessing whether the ~6% gain over self-consistency is cost-efficient.

- **The 9-point gap between oracle (82.79%) and selection agent (73.01%) is undiagnosed.** The paper emphasizes the oracle upper bound as proof of headroom but does not analyze *why* the selection agent fails on ~9% of cases where a correct candidate exists in the pool. Whether failures stem from schema ambiguity, SQL semantic equivalence issues, or specific SQL clause types is unknown. This diagnosis is directly actionable for future improvement.

---

## Nice-to-Haves

- **Compute-normalized comparison with self-consistency.** A comparison between CHASE-SQL and self-consistency given the same total number of LLM calls (rather than the same number of candidates per generator) would clarify whether gains stem from architectural improvements or simply from greater test-time compute budget.

- **Failure mode analysis for 258 unresolved questions.** Fig. 3a shows 258 questions where no generator produces a correct candidate. Characterizing these by difficulty level, SQL features, or domain would help the community understand CHASE-SQL's current limits and guide future work.

- **Query Fixer sensitivity analysis.** β=3 is set without justification; a brief ablation over β∈{1,3,5} would confirm this is not a critical hyperparameter.

- **Selection agent training data statistics.** The number of pairwise training examples, their correct/incorrect distribution, and any class-imbalance handling are unreported, making the selection agent's training difficult to reproduce.

- **Generator-level semantic diversity quantification.** A measurement of how often the 21 candidates (7 per generator) produce semantically distinct execution results (beyond syntactic variation) would strengthen the "diversity" claim and clarify when the pairwise selection adds value over simple deduplication.

- **Releasing model weights and fine-tuning scripts.** Given that independent verification of a 73.0% leaderboard number is not possible from published code alone, releasing selection model weights and prompt templates would substantially increase trust in the result.

---

## Removed Points

*These points are flagged for removal. Treat them with caution — they were raised in sub-reviews but are factually incorrect, apply unfair standards, or constitute style nitpicks.*

- **"CHASE acronym unexpanded"** — Minor formatting nitpick with no bearing on scientific content. Removed per formatting/style rule.

- **"MCS-SQL outperforms CHASE-SQL on Spider (89.6% vs 87.6%)"** — MCS-SQL uses Spider training data; CHASE-SQL does not. The paper explicitly acknowledges this asymmetry ("placing it second among methods that have undergone specific training or prompt optimization for the Spider dataset"). The comparison is intentionally favorable to the baseline to demonstrate stronger generalization, not a weakness.

- **"Weak baseline in Table 4 inflates gains"** — The stated purpose of Table 4's baseline ("original BIRD prompt + zero-shot CoT") is to measure the *isolated* contribution of each new CoT strategy, not to compare against other full systems. Comparing against CHESS's full pipeline would conflate the contribution of other CHESS components. The choice is methodologically appropriate.

- **Statistical significance / confidence intervals** — Single-run evaluation on BIRD and Spider is the established norm for this benchmark community; requesting bootstrap CIs is not a standard expectation and is removed per community standards rule.

- **"Computational complexity of pairwise comparisons is too high"** — While the cost concern is kept as a weakness (uncharacterized cost), the specific criticism that 420 comparisons *by itself* is intractable or disqualifying is not substantiated; the paper should quantify the cost but the design is not inherently unreasonable.

- **"Contributions should be in a bulleted list"** — Style nitpick; contributions are clearly described in the introduction narrative.

- **"Why exactly three generators and not others?"** — The paper provides empirical justification (Venn diagram, complementarity analysis in Fig. 3). Demanding a formal theoretical justification for an empirical systems paper exceeds field norms.

---

## Novel Insights

The most genuinely novel observation synthesized across the reviews is the interaction between **lower-bound performance and selection agent returns**: Fig. 2 reveals that OS ICL has a higher lower bound than the two CoT methods, meaning more of its candidates are uniformly correct, which paradoxically limits the marginal benefit of sophisticated selection for OS ICL compared to CoT methods. This suggests that the optimal strategy for future selection-based systems should explicitly trade off *lower bound diversity* (ensuring not all candidates collapse to the same wrong answer) against *quality* (minimizing invalid candidates). The finding implies that measuring only upper-bound oracle performance is insufficient to predict whether a selection agent will effectively recover gains—the lower bound is an equally important diagnostic. This has implications beyond Text-to-SQL for any system combining diverse candidate generation with learned selection.

---

## Suggestions

1. **Fix the "single LLM call" claim in Section 2.3.** Revise the description of Algorithm 1 to accurately state the number of sequential LLM calls; consider quantifying the average number of sub-questions produced to give readers a practical sense of the call overhead.

2. **Clarify the Query Plan CoT mechanism unambiguously.** Add a sentence explicitly stating whether the query plan is (a) generated synthetically by the LLM as a reasoning format, or (b) obtained from executing EXPLAIN on a preliminary SQL draft. If (b), describe how the initial draft is obtained to resolve the bootstrapping concern.

3. **Assess or mitigate the GT hint train-test mismatch.** Re-evaluate binary selection accuracy on pairs generated *without* GT hints, or report a breakdown showing what fraction of training pairs relied on GT guidance. If a significant fraction relied on GT, retrain without this and report the delta.

4. **Add a cost analysis table.** Provide average LLM call counts and approximate API token consumption per query (perhaps in the appendix) for at least three operating points: single-generator no-selection, three-generator self-consistency, and full CHASE-SQL. This is essential for practitioners.

5. **Add a brief failure case study for the 258 unresolved questions.** Even a coarse breakdown by SQL feature category (e.g., nested aggregation, multi-hop joins) or difficulty level would substantially improve the paper's analytical contribution and guide future work.

6. **Report OS ICL synthetic example error rates.** A small-scale analysis (e.g., on 100 questions) measuring what fraction of synthesized examples contain SQL errors, and how generation quality changes when using validated vs. unvalidated examples, would address a real concern about the method's reliability.

---

### Evaluation on Key Axes

- **Novelty:** Moderate-to-good. The Query Plan CoT and OS ICL are concrete new contributions; pairwise selection via a fine-tuned binary classifier is adapted from preference optimization literature. The primary novelty is in the integration and the specific design of each component for Text-to-SQL.
- **Technical soundness:** Adequate, with notable issues. The Algorithm 1 inconsistency and QP CoT ambiguity are real problems; the GT hint train-test mismatch is a substantive methodological gap that could affect the reported selection accuracy.
- **Empirical support:** Strong for the system-level claims; the ~6% selection advantage over self-consistency is well-supported. Weaker for individual component design choices (e.g., LSH vs. alternatives, β=3 for the fixer, OS ICL example correctness).
- **Significance:** High. BIRD is the most competitive public Text-to-SQL benchmark; a ~6-point improvement over the prior best published method is a meaningful advance with practical implications.
- **Clarity:** Generally good; the framework description and algorithmic presentation are clear, but the QP CoT mechanism and Algorithm 1 contain the inconsistencies noted above.

MY FINAL SCORE: <pineapple>6.4</pineapple>