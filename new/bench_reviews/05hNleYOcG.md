Now I have all the information I need. Let me write the final review.

---

## Summary

PLAGUE introduces a modular, plug-and-play framework for multi-turn jailbreak attacks that decomposes the attack lifecycle into three phases—Planner, Primer, and Finisher—each of which can be independently swapped or enhanced. The framework incorporates backtracking (removing failed turns from the target's history while retaining them in the attacker's), reflection via rubric scoring, and a strategy-retrieval memory bank. Evaluated on five frontier models (OpenAI o3, o1, Claude Opus 4.1, Deepseek-R1, Llama 3.3-70B), PLAGUE achieves the highest attack success rates across the board, reporting up to 81.4% SRE on o3 and 67.3% on Claude Opus 4.1.

## Strengths

- **Three-phase decomposition with demonstrated incremental contributions**: The ablation in Table 3 shows each component (backtracking, reflection, planning, strategy retrieval) contributing incrementally to ASR on o3 (0.587 → 0.612 → 0.761 → 0.773 → 0.814). This provides genuine insight into which mechanisms drive multi-turn attack success.

- **Strong empirical results on contemporary frontier models**: Testing on o3, Claude Opus 4.1, and Deepseek-R1 is more up-to-date than most prior work, and the consistent improvements across all five models are noteworthy.

- **Plug-and-play modularity is concretely demonstrated**: Swapping GOAT for Crescendo as the Finisher for Claude Opus 4.1 yields substantial gains (Table 4: 0.465 → 0.673 SRE), and incorporating ActorBreaker's planner improves diversity by 15% (Figure 3). This is a practical and useful finding.

- **Backtracking mechanism is simple and effective**: Removing failed turns from the target's history while retaining them in the attacker's history addresses a real failure mode in multi-turn attacks, and the ablation shows it is the largest contributor on Claude Opus 4.1 (0.222 → 0.396 SRE).

## Weaknesses

### Fatal

None.

### Major

- **Factual error in headline improvement claim**: The paper states (Section 1, line 19; Section 5.1, line 132): "we outperform the previous best — GOAT by a factor of 32.14%" on o3. However, on o3, ActorBreaker achieves SRE of 0.616 while GOAT achieves 0.587 (Table 2), making ActorBreaker the previous best in SRE—not GOAT. The 32.14% figure corresponds to improvement over ActorBreaker (0.616 → 0.814, relative improvement ≈ 32.1%), not GOAT. The improvement over GOAT in SRE would be ~38.7%. This misattribution of the baseline from which the improvement is measured is a factual error in the paper's central narrative claim.

- **Modified baselines without reporting original performance numbers**: The paper modifies baselines for "apples-to-apples comparison" (Section 4): disabling GOAT's history mechanism, adding rubric scoring to GOAT, limiting ActorBreaker to K=2 actors, and capping Crescendo at 6 turns with backtracking counts removed. While the modifications are disclosed, the paper claims "the impact on GOAT's performance with and without an attack history is negligible" without providing any supporting data (no table, no numbers). Without original baseline numbers, the headline improvements cannot be independently verified—readers cannot tell whether PLAGUE outperforms the original baselines or only modified versions. Reporting both original and modified baseline numbers is essential for the claims to be credible.

### Minor

- **No variance or significance reporting**: All results are averaged over only 3 runs with no standard deviations, confidence intervals, or significance tests. Given the stochastic nature of LLM-based attacks and the small number of runs, the reliability of the reported improvements is uncertain—particularly for small ablation differences like the Planner slightly decreasing Bin-ASR from 0.59 to 0.582 on o3 (Table 3).

- **"Lifelong learning" terminology is oversold**: The mechanism is cosine-similarity retrieval from a growing vector database—standard RAG with an expanding retrieval store. No model weights are updated, no adaptation occurs beyond appending to a retrieval index, and the memory bank is initialized with only two strategies adapted from Crescendo. While the ablation shows RSS contributes (0.773 → 0.814 SRE on o3), calling this "lifelong learning" overclaims relative to the established meaning of the term.

- **Efficiency analysis incomplete for total compute cost**: Table 5 counts only target-model calls, evaluator calls, and planner calls. PLAGUE uses Deepseek-R1 as the attacker, Qwen3-235B as the evaluator, and makes additional calls for reflection and summarization—none of which are counted. The total computational cost of PLAGUE is likely substantially higher than GOAT (which has no evaluator or planner), but this is not reflected in the efficiency analysis.

- **Per-model component selection undermines "plug-and-play" simplicity**: On Claude Opus 4.1, PLAGUE with the GOAT Finisher achieves only 0.465 SRE—worse than base Crescendo (0.48). Achieving the headline 67.3% requires swapping to the Crescendo Finisher (Table 4), which itself requires experimentation to discover. The framework is plug-and-play in structure but requires per-model tuning in practice.

- **No analysis of strategy retrieval behavior**: The paper provides no data on how many unique strategies are discovered during evaluation, how often cosine similarity retrieval finds strategies above the 0.6 threshold versus falling back to random retrieval, or whether the memory bank grows meaningfully. This makes the "lifelong learning" component's actual contribution opaque.

### Trivial

- Duplicate ActorBreaker row in Table 2 (lines 108–109 are identical)—likely a formatting or layout artifact.

## Nice-to-Haves

- Sensitivity analysis on scoring thresholds (7/10, 3/10, 8/10) to show robustness of results to these ad hoc choices.
- Agreement analysis between the internal Rubric Scorer's success criterion (score > 8/10) and the final Judge evaluation to quantify false positive/negative rates.
- Attacker model ablation (e.g., running with a smaller attacker than Deepseek-R1) to disentangle framework contribution from attacker capability.
- Qualitative attack trajectories showing complete multi-turn conversations with scores and backtracking events.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Fair apples-to-apples comparison" as a strength** (from Strength Finder): This conflicts with the verified Major weakness about modified baselines without original numbers. The modifications are disclosed, but claiming the comparison is "fair" or "apples-to-apples" without evidence that modifications don't significantly alter baseline performance is unsupported.

- **Criticisms about ASR@K=2 compute asymmetry giving PLAGUE 2× total compute** (from Harsh Critic): On closer reading, the paper defines the budget as 6 target-model calls per attempt, and ASR@K=2 means 2 independent attempts for all methods. Both PLAGUE and ActorBreaker get 2 × 6 = 12 target-model calls total. The target-model budget is fair. The concern about attacker-side compute is valid but already captured in the "efficiency analysis incomplete" minor weakness.

- **Duplicate ActorBreaker rows as a significant error** (from Harsh Critic): This is at most a trivial formatting artifact, not a meaningful data error.

- **Claude Opus 4.1 results being "split across tables" as a weakness** (from Harsh Critic): The paper explicitly notes "*Best results for Claude Opus 4.1 are in Table 4" (line 114) and explains the model-specific Finisher swap in Section 5.1. This is by design, not obfuscation.

- **Concerns about unreleased/unavailable baselines or reproducibility** (implicit in some criticisms): Per the review rules, if the paper cites it, it exists.

## Novel Insights

The ablation in Table 3 reveals a genuinely interesting finding: different defense mechanisms in target models are vulnerable to different attack components. Reflection is the largest contributor for o3 (0.612 → 0.761 SRE), while backtracking is most critical for Claude Opus 4.1 (0.222 → 0.396 SRE). This suggests that o3's safety is more susceptible to iterative query refinement, while Claude's safety relies more on conversation-context integrity that backtracking disrupts. This model-specific vulnerability profile is a useful insight for both attackers and defenders.

## Suggestions

- Correct the 32.14% claim: either attribute the improvement to the correct baseline (ActorBreaker in SRE on o3), or compute and report the correct improvement over GOAT (~38.7% in SRE, ~48.8% in Bin-ASR).
- Report original (unmodified) baseline numbers alongside the modified versions, especially for GOAT where the claim of "negligible" impact is made without evidence.
- Add standard deviations to Table 2 and Table 3 results, even with just 3 runs, to allow readers to assess the reliability of differences.

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| LLMs Get Lost In Multi-Turn Conversation | VKGTGGcwl6 | 8.0 | Far stronger methodology and analysis; PLAGUE is below this due to comparison fairness issues and factual errors |
| ARMs: Adaptive Red-Teaming Agent | wQ4OykcxaV | 5.5 | Very similar paper (plug-and-play attack framework, strategy memory); PLAGUE has comparable strengths but worse comparison fairness issues |
| SEMA: Multi-Turn Jailbreak | 6eSNG1VNkl | 5.0 | Similar topic; PLAGUE has stronger empirical coverage but SEMA avoids baseline modification issues |
| GRAF: Multi-Turn Jailbreaking | f9BuANYtJf | 4.4 | Had fundamental methodology issues (fabricating responses); PLAGUE's issues are less severe |
| Emoji Game | e8b3fRYeDt | 0.67 | Far weaker paper; PLAGUE is clearly above this |

PLAGUE is comparable to ARMs (5.5) and SEMA (5.0) in its contribution profile—a modular framework with real empirical results—but has more significant issues with comparison fairness and a factual error in its headline claim. It sits below ARMs because the baseline modification issue directly undermines the paper's core claims. It sits above GRAF because PLAGUE's issues don't rise to the level of fundamentally flawed methodology. I place it at 5.0.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>