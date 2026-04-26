Now I have enough information to write the consolidated review. Let me carefully verify each claimed weakness against the paper text.

## Summary

BIRD-INTERACT introduces a dynamic, multi-turn text-to-SQL benchmark covering the full CRUD spectrum with 900 tasks, a function-driven user simulator to avoid ground-truth leakage, and two evaluation settings (c-Interact for protocol-guided interaction, a-Interact for agentic interaction). The benchmark reveals that state-of-the-art models achieve at most ~25% success rate, and a memory grafting experiment demonstrates that GPT-5's poor c-Interact performance stems from deficient interactive communication rather than weak SQL generation capability.

## Strengths

- **Genuine and important gap identification**: The paper correctly identifies that existing multi-turn text-to-SQL benchmarks use static conversation transcripts and restrict evaluation to SELECT-only operations. BIRD-INTERACT addresses both simultaneously with 900 tasks spanning full CRUD operations (190 DM tasks in the FULL set alone), making it the most operationally diverse text-to-SQL benchmark to date.

- **Function-driven user simulator with quantified improvements**: The two-stage architecture (semantic parsing into AMB/LOC/UNA actions, then controlled response generation) addresses a documented failure mode of LLM-based simulators. Figure 6 shows concrete reliability improvement: failure rates on unanswerable questions drop from up to 67.4% (baseline) to 2.7% (function-driven), and Table 3 shows human alignment improves from 0.61 to 0.84 Pearson correlation (GPT-4o backbone) and 0.54 to 0.79 (Gemini backbone).

- **Dual evaluation settings reveal genuinely distinct capability profiles**: c-Interact and a-Interact are not redundant difficulty levels—they expose orthogonal strengths. GPT-5 ranks worst in c-Interact priority-question SR (14.50%) but best in a-Interact (29.17%), a 2× swing that demonstrates the two settings evaluate fundamentally different capabilities.

- **Memory grafting experiment separates communication from generation**: By providing GPT-5 with gold-standard clarification histories from better-performing models, Figure 5 shows substantial performance improvement, directly demonstrating that the bottleneck is interaction strategy rather than SQL generation. This is a novel and informative diagnostic technique.

- **Systematic ambiguity injection**: The three-type ambiguity injection methodology (superficial, knowledge chain breaking, environmental) and the state-dependent follow-up sub-tasks requiring reasoning over modified database states are thoughtful design choices that increase ecological validity.

## Weaknesses

### Fatal
None.

### Major

- **The "ITS Law" is overclaimed relative to its evidence**: Section 5.2 proposes an "ITS Law" defined as: "A model satisfies this law if, given enough interactive turns, its performance can match or even surpass that of the idealized single-turn task." The empirical support consists of Figure 4, showing scaling behavior primarily for one model (Claude-3.7-Sonnet) on the LITE subset. The paper does not systematically test whether other models satisfy this condition, does not establish boundary conditions, and does not compare to theoretical baselines. The condition is also trivially satisfiable under any error-correcting regime (more attempts → more chances to succeed), making "law" a mischaracterization of what is currently a single-model empirical observation. This matters because elevating a preliminary pattern to a "law" misrepresents the state of evidence and could misdirect future work.

- **Causal attribution from memory grafting is overstated**: Section 5.2 concludes GPT-5's poor c-Interact performance "stems from a deficiency in its interactive communication abilities rather than its core generation capability." The memory grafting experiment provides GPT-5 with gold-standard clarification information from other models' successful interactions and shows improvement—but this demonstrates that better *input information* improves output quality, not that the *process of eliciting* that information is the specific bottleneck. Alternative explanations (poor instruction-following for the c-Interact protocol, poorly calibrated prompting for the constrained dialogue format) are not ruled out. The finding is genuinely informative; the specific causal claim about "communication deficiency" is overstated. This matters because targeting the right bottleneck determines the direction of future improvement efforts.

### Minor

- **Human alignment validation is underpowered**: The Pearson correlations in Table 3 are computed across n=7 models, making the p=0.02 for r=0.84 uninformative—any monotonic relationship over 7 points can appear significant. Cross-task correlation (n=100 tasks) would be a more informative validation of whether the simulator produces human-like success/failure patterns. That said, the relative improvement (0.84 vs 0.61 for GPT-4o; 0.79 vs 0.54 for Gemini) between function-driven and baseline simulators is informative regardless of the absolute p-value.

- **Budget parameters lack sensitivity analysis**: The budget formulas (B_base=6, coefficient of 2 on m_amb in a-Interact vs. 1 in c-Interact) are design choices presented without justification or sensitivity analysis. Whether model rankings are robust to these specific parameter values is not tested, which could matter given that these parameters directly affect turn budgets and therefore performance.

- **Single-run evaluation without variance**: The paper acknowledges conducting single runs due to cost (Section 5). For binary outcomes over 600 tasks at success rates of ~8–25%, standard errors are ~1.1–1.8%, so differences of 2–3% between models may not be statistically significant. While the larger patterns (GPT-5's cross-mode swing, the separation between top and bottom models) are robust, smaller ranking differences should be interpreted with caution.

### Trivial
None.

## Nice-to-Haves

- Per-ambiguity-type resolution rates (superficial vs. knowledge chain vs. environmental), which would inform which ambiguity types are hardest to resolve and increase the benchmark's practical value.
- Sensitivity analysis on budget parameters to confirm robustness of model rankings.
- Multi-run evaluation on a subset (e.g., LITE) with confidence intervals to establish statistical reliability.

## Removed Points

- **Simulator limited to 3 action categories**: The harsh critic argued the AMB/LOC/UNA categorization cannot capture novel clarification types. This is a real limitation but is also a deliberate design tradeoff for controllability and evaluation validity. The paper should discuss it more explicitly, but it does not undermine the benchmark's contribution since unconstrained simulators have known failure modes (ground-truth leakage, task deviation) that this design specifically addresses.

- **c-Interact vs. a-Interact confounds prevent causal conclusions**: The harsh critic argued budget formulas, action spaces, and evaluation protocols differ across settings, confounding comparison. However, the paper's key claim is about the *cross-model pattern* (the same model performing differently across modes), not about direct A/B comparison of absolute performance between modes. The dramatic GPT-5 swing from worst to best across modes is robust to these confounds.

- **Reproducibility concerns (single-run, hyperparameters)**: These are trivial implementation details. The paper documents all experimental settings clearly (temperature=0, user patience=3, B_base=6) and will release code, databases, prompts, and interaction logs. Standard practice for large-scale LLM benchmarking.

- **Missing appendix/references**: Parser artifacts, not paper issues.

- **Formatting issues**: Parser artifacts.

- **Demand for larger datasets or more models**: The 900-task suite across 7 models is already substantial for the benchmark's scope.

- **Demand for theoretical proofs**: This is an empirical benchmark paper; theoretical proofs of ITS scaling are not expected in this venue.

- **Claims that cited models/tools don't exist**: Per hard rules, all cited entities are treated as existing.

- **Demand for error taxonomy of interaction failures**: Would strengthen the paper but is beyond its stated scope of benchmark construction and initial evaluation.

## Novel Insights

The dual evaluation setting (c-Interact vs. a-Interact) reveals a striking capability inversion—GPT-5 ranks worst in protocol-guided interaction but best in agentic interaction—suggesting that current frontier models may possess strong autonomous planning abilities while being surprisingly poor at structured, protocol-governed dialogue. This has implications for deployment: the same model can be the best or worst choice depending on the interaction paradigm, a dimension of evaluation that existing benchmarks entirely miss.

## Suggestions

- Replace the "ITS Law" framing with "ITS Scaling" or "ITS Hypothesis," and explicitly characterize it as a conjecture supported by preliminary evidence requiring validation across more models and boundary conditions.
- Temper the memory grafting conclusion from "deficiency in interactive communication abilities" to "the bottleneck appears to be in acquiring task-relevant information through the interaction protocol, rather than in applying that information to SQL generation"—which accurately reflects what the experiment demonstrates.
- Report the per-task (n=100) human correlation alongside the per-model (n=7) correlation to strengthen the human alignment validation.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Spider 2.0 | XmProj9cPs.md | 8.0 | Similar domain (text-to-SQL benchmark), but Spider 2.0 has no overclaimed analytical conclusions. BIRD-INTERACT is comparable in contribution breadth but weaker in analytical rigor. |
| WildBench | MKEHCx25xp.md | 7.33 | Strong benchmark with novel metrics and high validation correlation. BIRD-INTERACT has less rigorous validation (n=7) but broader scope. |
| GameArena | SeQ8l8xo1r.md | 6.5 | Interactive benchmark with human evaluation. BIRD-INTERACT has similar innovation scope with some overclaiming. |
| LAM Simulator | Dpqw0namg3.md | 6.0 | Simulator-based framework, overclaimed analysis, rejected. BIRD-INTERACT is stronger in benchmark construction. |
| Proactive Agent | sRIU6k2TcU.md | 5.5 | Simulator-based evaluation, overclaimed conclusions. BIRD-INTERACT has more substantial contributions. |
| PingPong | 996aKQIom0.md | 3.83 | User simulation benchmark with poorly validated evaluation. BIRD-INTERACT is significantly stronger in every dimension. |

BIRD-INTERACT sits above the medium-quality anchors (LAM Simulator at 6.0, Proactive Agent at 5.5) because its benchmark construction is genuinely comprehensive and its contributions (function-driven simulator, CRUD coverage, dual settings, memory grafting) are real and valuable. It sits below the high-quality anchors (Spider 2.0 at 8.0, WildBench at 7.33) because the analytical claims (ITS "Law," communication deficiency attribution) outpace the evidence, and the human alignment validation is underpowered. It's comparable to GameArena (6.5) which also introduces an interactive evaluation framework with novel methodology but with some analytical concerns.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>