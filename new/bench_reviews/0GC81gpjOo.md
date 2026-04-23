## Summary

The paper investigates the relationship between Theory of Mind (ToM) levels and cooperative trends in LLM-based multi-agent systems, finding that higher-level ToM agents do not necessarily exhibit better cooperative trends. It proposes a stable coalition matching mechanism that leverages belief-action alignment (derived from ToM reasoning) to form coalitions, along with an adaptation for specialized agent abilities, demonstrating improved cooperative trends and task performance across programming, debate, and reasoning tasks.

## Strengths

- **Interesting and novel research question**: The finding that higher-order ToM reasoning may not straightforwardly improve cooperation in LLM agents is genuinely counterintuitive and worth investigating. This challenges the common assumption that more cognitive capability always leads to better collaboration (Section 3, Table 1).

- **Clean recursive ToM formulation**: Equation (1) provides a well-defined recursive belief function for k-level ToM, giving a principled formal framework that could be reused in future multi-agent LLM work (Section 4.1).

- **Multi-model evaluation**: Testing across five different LLM backbones (GPT-3.5, GLM-4, Llama-3-70B, Gemini-1.5-flash, Claude-3-sonnet) provides breadth and some evidence of generality (Section 6.1).

- **Matching mechanism shows promising empirical trends**: Table 2 demonstrates that stable matching improves FTM scores for both 1-ToM and 2-ToM agents, with 2-ToM agents surpassing 1-ToM by round 5 in several models (e.g., GLM-4: 91.0 vs 84.0; Claude-3-sonnet: 95.45 vs 91.07) (Section 6.3).

- **Task performance improvements over baseline**: Table 3 shows 2-ToM with matching achieves 90.0% Pass@1 on HUMANEVAL and 90.4% on MBPP, outperforming MetaGPT (85.4%, 86.5%), confirming that cooperative trend improvements translate to task performance (Section 6.3).

## Weaknesses

### Fatal
None.

### Major

- **Sign error in the specialized ability adaptation (Section 5.2)**: The base preference ordering defines lower B_i(S) as preferred (S₁ ≻_i S₂ ⟺ B_i(S₁) < B_i(S₂), line 123). The adaptation adds a positive term: B'_i(S) = B_i(S) + λ·(1/|S|)Σα_j, where "higher values of α_i indicate greater specialized ability" (line 173). Since lower B' is preferred (line 177), adding positive α_j *penalizes* coalitions with more capable agents—the exact opposite of the paper's claim that "the stable matching algorithm will prioritize agents with higher specialized abilities" (line 185). This is not a notation issue; the entire section's logic depends on the sign being correct. The convergence/stability proofs in Appendix G referenced for this adaptation would also need re-derivation. Since λ=1 is the default evaluation parameter, if this adaptation is used in the experiments, the results may not reflect what the paper claims they do.

- **The FTM metric conflates belief prediction accuracy with cooperative behavior**: The paper explicitly defines "cooperative trends as the tendency of agents to exhibit accurate predictions about their teammates' actions" (line 27), but the paper's own motivation (citing Ridinger & McBride, 2017) acknowledges that ToM alone is insufficient and that "agents may also need to be willing to positively reciprocate and cooperate with others" (line 29). A perfectly selfish agent could predict teammates' actions perfectly and still act against the team's interest. FTM measures epistemic alignment, not cooperative disposition. This conflation means the paper's headline narrative ("higher ToM → worse cooperation, matching fixes it") is built on a metric that does not measure cooperation as commonly understood. The paper should either redefine the metric to capture actual cooperative behavior or substantially reframe its claims.

- **"With matching" improvements confound coalition selection with the ToM-based mechanism**: The "without matching" baseline forces all agents to work together; "with matching" selects a preferred subset. Any improvement could be due to simple teammate selection (filtering out poorly-matched agents) rather than the ToM-based belief alignment mechanism specifically. No ablation tests a non-ToM-based matching strategy (e.g., random matching, similarity-based matching without ToM beliefs, or matching based on past task performance). Without this, the paper cannot attribute improvements to the ToM mechanism rather than to coalition selection in general (Section 6.3, Tables 2–3).

### Minor

- **Table 1 motivating finding is not uniformly supported by the data**: The table title claims "Low ToM agents show Higher cooperative trends," but several cells show the opposite: HUMANEVAL with Gemini-1.5-flash at R=1 (2-ToM: 80.56 > 1-ToM: 75.0); MBPP with GLM-4 at R=5 (2-ToM: 86.3 > 1-ToM: 85.2); MBPP with Llama-3-70B at R=1 (2-ToM: 81.7 > 1-ToM: 81.3); MBPP with Claude-3-sonnet at R=5 (2-ToM: 54.4 > 1-ToM: 48.6). While the majority of cells support the claim, the universal assertion in the table title is an overstatement given these counterexamples.

- **Self-evaluated alignment scores may introduce systematic bias**: The alignment measure φ(bᵢᵏ(aⱼ) − âⱼ) is computed by prompting the agent to self-evaluate (line 127). LLM self-evaluation is known to be unreliable and potentially systematically biased. More critically, the self-evaluation prompt complexity differs between 1-ToM and 2-ToM agents, making it plausible that FTM differences could partially reflect prompt-induced self-evaluation bias rather than genuine differences in predictive accuracy. No external/ground-truth alignment measure validates the FTM findings.

- **No variance or significance tests reported**: All results are presented as point estimates with no standard deviations or confidence intervals. This is especially concerning for the debate experiment with N=11 repetitions, where win rates of 65.45% vs. 61.82% vs. 67.27% are not distinguishable at this sample size (Section 6.4, Table 4).

- **Algorithm 1 underspecifies the stable matching computation**: Line 8 says "Update stable coalition S based on preference orders" without specifying the algorithm. Standard Gale-Shapley applies to two-sided matching; the mechanism for computing stable coalitions from one-sided preferences is not detailed in the main text (Section 5.1).

### Trivial
None.

## Nice-to-Haves

- An ablation with a non-ToM matching baseline (e.g., random coalition formation or past-performance-based matching) to isolate the contribution of the ToM-based belief alignment mechanism.
- An external validation of FTM scores using objective similarity measures (e.g., embedding similarity between predicted and actual action text) to complement self-evaluation.
- Variance and statistical significance across multiple runs, especially for the debate experiment.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"0-level ToM is not really ToM"**: The critic argues that what the paper calls 0-ToM (just recording history) is not ToM and that the numbering is misleading. While the observation is valid that 0-ToM here corresponds to no mental state attribution, the paper follows existing literature conventions (De Weerd et al., 2015; Li et al., 2023c cited in line 81) for its level numbering. This is a terminology preference, not a substantive error. **Removed as a terminology nitpick that doesn't affect the paper's substance.**

- **"NP-hardness claim vs. trivial experimental setting"**: The critic notes that with N=4 agents, the matching problem is trivially just pairing, yet the paper claims NP-hardness in limitations. The NP-hardness claim is about the general problem, not the experimental setting. The experimental setting is a proof-of-concept demonstration. **Removed as scope creep — the limitation discussion about general complexity is valid even if the experimental setting is small.**

- **"The ChatEval w. ToM and DyLAN w. ToM baselines are non-standard"**: The critic argues these are not standard configurations. The paper describes them in Section 6.5 as existing frameworks with ToM capabilities integrated. **Removed — this is a concern about not-yet-released configurations, which per rules we treat as existing if cited.**

- **"The 'human-like' claim is unsupported"**: The critic says the "human-like" claim in the abstract rests only on a single citation. The paper references Ridinger & McBride (2017) to ground the cognitive insight, and the claim is about the *potential* to create human-like strategies, not that the current system is validated as human-like. **Removed as an over-interpretation of a motivational claim.**

- **"Missing related works"**: The critic implies missing references. **Removed per hard rules — we cannot confirm existence of uncited works.**

- **"Overthinking explanation is post-hoc and untested"**: The paper offers "overthinking" as an intuitive explanation for the empirical finding, not as a proven mechanism. The qualitative case study in Section 6.4 provides some supporting evidence. While a formal test would strengthen this, the current treatment is reasonable as a discussion point. **Weakened to minor — the explanation is speculative but acknowledged as intuitive rather than proven.**

- **Strength finder's claim about "1-ToM agents consistently achieve higher FTM scores than 2-ToM agents across all five LLM models"**: This is factually incorrect — Table 1 shows counterexamples as documented above. **Removed as conflicting with verified weakness about Table 1 inconsistencies.**

- **Strength finder's claim about "adaptation for specialized abilities increases practical applicability"**: Given the sign error in this very adaptation, this strength is unreliable. **Removed because it conflicts with a verified Major weakness.**

## Novel Insights

The paper's most interesting observation — that LLM agents with higher-order recursive reasoning (2-ToM) may actually perform worse at predicting teammates' actions than simpler 1-ToM agents — resonates with known phenomena in game theory where higher-order reasoning can lead to "over-thinking" and departures from equilibrium. However, the evidence for this is mixed across models/benchmarks and the metric used (self-evaluated belief prediction) may not capture genuine cooperative dynamics. The idea that a stable matching mechanism can recover the value of higher-order reasoning by selecting compatible teammates is promising, but the current work cannot cleanly separate the benefit of teammate selection from the benefit of ToM-based alignment specifically.

## Suggestions

- Fix the sign error in Section 5.2: the adaptation should subtract the ability term (B'_i(S) = B_i(S) − λ · (1/|S|)Σα_j) or equivalently reverse the preference direction for the ability component. Verify whether experiments used the erroneous sign and re-run if so.
- Add at least one non-ToM matching baseline (e.g., random coalition selection) to demonstrate that the improvements are attributable to the ToM-based belief alignment and not merely to any team selection procedure.
- Consider renaming FTM or adding a separate metric that captures actual cooperative behavior (e.g., task performance improvement attributable to coordination, or willingness to defer/adjust actions based on teammates), to avoid conflating prediction accuracy with cooperation.
- Report standard deviations across multiple runs and, where sample sizes are small (e.g., debate N=11), include confidence intervals or significance tests.

## Score and Decision

**Calibration anchors compared:**

| Paper | Avg Score | Decision | Comparison |
|-------|-----------|----------|------------|
| st77ShxP1K (Conformity in LLM multi-agent) | 7.5 | Accept (Oral) | This paper is substantially weaker: less rigorous benchmarking, no ablation study, sign error in formulation |
| otW0TJOUYF (ToM for multi-agent tasks) | 6.75 | Accept (Poster) | This paper has a cleaner ToM module with ablations; our paper lacks comparable experimental rigor |
| cfL8zApofK (LLM negotiation game) | 4.75 | Reject | Comparable: limited novelty, variance issues, single-model concerns; our paper tests more models but has the sign error |
| ueqTjOcuLc (Social psychology LLM collaboration) | 5.0 | Reject | Comparable: variance issues, conceptual framing concerns; our paper has more models but the FTM metric conflation is worse |
| 5nldnvvHfw (Sign error in optimization) | 2.5 | Reject | Our paper's sign error is in an extension, not the core theorem, so less severe |

This paper sits between the medium-scoring rejected papers (4.75–5.0) and the low-scoring math-error papers (2.5). The sign error in Section 5.2 is less severe than those in the 2.5-scored papers (which had errors in core theorems), but the combination of the sign error, the FTM metric conflation, missing ablations, and inconsistent motivating data places it solidly below acceptance. It has more substance than the weakest papers but the cumulative weaknesses undermine confidence in the central claims. I place it slightly below the medium rejected papers because of the sign error and metric conflation, which go beyond the typical "overclaimed results" pattern of the 4.75–5.0 papers.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>