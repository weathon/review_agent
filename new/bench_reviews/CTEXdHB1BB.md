Now let me look at the Figure 3 data more carefully and then run calibration searches.Now I have enough information to write the final consolidated review. Let me also verify the Figure 3 data consistency issue one more time.Now I have all the data I need to write the final review. Let me synthesize everything.

---

## Summary
CANON (Conditional AdvaNtage estimatiON) proposes a framework for RLVR training of large language models that regroups sampled responses by a target metric (entropy or response length) into two equal-sized groups, then computes inter-group advantages (cross-group comparison) and intra-group advantages (within-group comparison). The key theoretical insight is that DR.GRPO is a special case of this framework (μ=0.5 in Eq. 5) with equal weighting. Empirically, CANON-Inter improves math reasoning (+1.9 points over DR.GRPO), CANON-Intra improves high-complexity logic reasoning (+5.2 points on XLarge subset), and CANON-Eff (length-weighted variant) achieves a superior Pareto frontier for performance-efficiency trade-off.

---

## Strengths

- **Novel advantage decomposition with clean unification:** CANON's inter/intra group framework is elegant, and the proof that DR.GRPO equals the equal-weight (μ=0.5) case (Eq. 7) is non-trivial and theoretically satisfying. This makes CANON's contribution clearly positioned relative to existing work.

- **Complementary empirical roles of CANON-Inter and CANON-Intra:** Table 1 shows distinct and interpretable behaviors — CANON-Inter (entropy) improves math reasoning by 1.9 points (55.7→57.6) while CANON-Intra (entropy) achieves a 5.2-point gain on the hardest ZebraLogic subset (XLarge: 15.1→20.3). The differentiation is principled and well-supported by training dynamics (Figure 2d–f), including the "reflection gain" analysis that correlates intra-group advantage with effective exploration behavior.

- **Mechanistic interpretability via μ sweep (Figure 5):** The hierarchical entropy trend across seven values of μ (0.0–1.0) provides concrete evidence that CANON provides continuous, directional control over the target metric, without pre-specifying direction. This is a genuine empirical validation of Theorem 2's selective amplification claim.

- **Superior Pareto frontier for efficiency (Figure 4c):** The multi-α Pareto frontier comparison is methodologically sound and shows CANON-Eff dominating all baseline efficiency methods across operating points, including Length Reward (+) which collapses catastrophically (54.8→22.5 accuracy) with a small coefficient change, while CANON-Eff remains stable.

- **Ablation confirms the mechanism (Table 4):** Direct numerical amplification (A=A×2) improves math only marginally (+0.4) but hurts logic (-1.1), while CANON-Intra improves logic by +2.9 and CANON-Inter improves math by +1.9. This supports the selective amplification argument.

---

## Weaknesses

### Fatal
None.

### Major

- **Figure 3's embedded data table contains values that are systematically inconsistent with Tables 1 and 2 — and the misassignment favors the paper's conclusion.** Cross-checking the table in Figure 3 against Tables 1 and 2:
  - *Llama-8B:* Figure 3 lists DR.GRPO as (Math=22.6, Logic=18.9). Table 2 shows DR.GRPO for Llama-8B is (Math Acc=22.0, Logic Acc=14.9). The values 22.6 and 18.9 correspond exactly to the *Cosin-First-Inter-Later-Intra* CANON-Dynamic result in Table 2.
  - *Qwen-1.5B:* Figure 3 lists DR.GRPO as (Math=46.8, Logic=17.0). Table 2 shows actual DR.GRPO for Qwen-1.5B as (Math Acc=46.4, Logic Acc=12.8). The values 46.8 and 17.0 correspond exactly to the *First-Inter-Later-Intra* CANON-Dynamic result in Table 2.
  - *Qwen-7B:* Figure 3 lists DR.GRPO as (Math=57.6, Logic=39.2). Table 1 shows actual DR.GRPO for Qwen-7B as (Math Acc=55.7, Logic Acc=26.2). Neither value matches; 57.6 corresponds to CANON-Inter (Entropy) in Table 1, and 39.2 is the Mid (not overall) logic score for DR.GRPO.
  - Meanwhile, CANON-Dynamic, CANON-Inter, and CANON-Intra values in Figure 3 are clearly schematic (perfectly symmetric: e.g., CANON-Inter=(35.2, 15.0), CANON-Intra=(15.0, 35.2), CANON-Dynamic=(35.2, 35.2)) and match no actual row in Tables 1 or 2.
  
  The effect is that the "DR.GRPO" rows in Figure 3's table display the *actual* CANON-Dynamic experimental numbers from Table 2, making DR.GRPO appear better on the radar chart than it actually is, while CANON-Dynamic is shown with artificial values. This makes the visual argument for CANON-Dynamic's superiority self-referential and misleading. The paper's authors should either: (a) clearly label Figure 3 as a conceptual schematic with illustrative values, or (b) replace it with actual experimental values from Table 2 on a common scale. This issue must be resolved before publication because Figure 3 is the primary visual argument for cross-model generalization.

- **CANON-Dynamic requires model-specific post-hoc strategy selection with no principled selection criterion.** Section 5.2 discloses that four scheduling strategies were tried (*First-Inter-Later-Intra*, *First-Intra-Later-Inter*, *Cosin-First-Inter-Later-Intra*, *Cosin-First-Intra-Later-Inter*) and the best-performing one was selected per model: Cosine for Qwen-7B and Llama-8B, accuracy-based for Qwen-1.5B. The paper acknowledges this explicitly ("A specifically designed strategy is acceptable for better performance in practice"), but this means the headline cross-model generalization claim is backed by a best-of-four search per model without a held-out validation set. The text claim that *First-Inter-Later-Intra* "consistently performs better than DR.GRPO across three models" (Section 5.2) further misleads — for Qwen-1.5B, this strategy underperforms DR.GRPO on Olympiad (42.4 vs. 43.9) and GSM8k (83.3 vs. 84.3). The overall Acc margin for Qwen-1.5B is only 46.8 vs. 46.4, which is within noise on small benchmarks. While the actual results in Table 2 do show net positive outcomes for each model's selected strategy, the claim of *consistent* generalizability of CANON-Dynamic without model-specific tuning is not supported.

### Minor

- **Theorem 1 is mathematically valid but its training implication is unestablished.** Theorem 1 proves the inter-group advantage ratio |Â_inter|/|Â_DR.GRPO| > 1 iff the groups are equal-sized. This justifies the 50/50 split design, but does not prove that a larger advantage magnitude *ratio* leads to better policy gradient updates or faster convergence. The step from "larger advantage signal for the conditioned metric" to "better training outcomes" is assumed, not proven. For Theorem 2, the independence assumption between conditions (entropy and length are stated as examples) may not hold in practice — entropy and response length are correlated. These are not fatal flaws (theoretical proof of advantage magnitude → training benefit is not standard in this field), but they are gaps worth flagging.

- **Budget-performance evaluation uses post-hoc response truncation rather than hard generation limits.** The budget-performance curves in Figure 4a/b truncate responses at various budget fractions after generation. Truncating a full-length response is not equivalent to generating under a hard length constraint — truncated responses may have incomplete reasoning chains. This is non-standard and the efficiency claims (2.63× performance at low token budget) should be validated under generation-time length limits.

- **α-weighting in Eq. 9 breaks the theoretical framing.** When α≠1, DR.GRPO is no longer a special case of CANON as stated in Eq. 7, and Theorem 1's justification for the inter-group advantage doesn't directly apply to CANON-Eff. The extension to weighted conditions is introduced purely empirically, which is fine, but the authors should be explicit that CANON-Eff is an extension beyond the theoretical framework.

### Trivial
None beyond the Figure 3 labeling issues already discussed at the Major level.

---

## Nice-to-Haves

- Replicate key results across multiple seeds with variance estimates, particularly for small benchmarks (AIME24/25 have 30 questions; the paper reports Avg@10 but no confidence intervals). The margins between CANON variants and DR.GRPO on AIME are 1–5 points, which could plausibly be within noise.
- Ablate the 50/50 group split ratio empirically (e.g., 30/70, 40/60) to validate the Theorem 1 recommendation, since the theorem justifies it in terms of advantage ratio but not training outcomes directly.
- A principled, online criterion for μ scheduling (e.g., adapting μ based on the observed advantage gap between groups during training) would make CANON-Dynamic practically useful without model-specific tuning.
- Qualitative response examples from CANON-Inter vs. CANON-Intra on representative ZebraLogic problems would strengthen the mechanistic story in Section 6.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic: "Abstract's 'direction-free' claim overstates the case."** The method is clearly described as adaptive (using current rollout data), not hand-coded. The Abstract says CANON "amplifies the impact of the target metric without presuming its direction" — this accurately distinguishes CANON from prior methods that require specifying "higher-is-better" or "lower-is-better" priors. This is a presentation nitpick that mischaracterizes the actual distinction being made. Removed.

- **Harsh Critic: "Entropy-based vs. length-based baselines are from different method dimensions."** The critic argues Section 5.1's comparison conflates advantage framework differences with length-signal differences. However, the paper evaluates CANON-entropy vs. entropy baselines, and CANON-length vs. length baselines, separately. The comparison is not mixed across metrics. This misreads the experimental design. Removed.

- **Harsh Critic: "Theorem 2 independence assumption fails for entropy-length correlation."** Theorem 2 states selective amplification holds for *independent* conditions. The paper uses this theorem to explain why grouping by entropy doesn't amplify length effects and vice versa. The harsh critic notes entropy and length are correlated in practice, which is true but irrelevant: Theorem 2 is stated about independent conditions as a formal property, and the experiments show the qualitative behavior (Table 4, Figure 5). The theorem is appropriately scoped. The practical correlation does weaken the formal guarantee, but this is a minor nuance, not a structural flaw. Moved to nice-to-have territory; kept as Minor weakness (Theorem 1/2 limitations).

- **Strength Finder: "Consistent empirical improvements across models and tasks (Table 2)."** This strength conflicts with the verified weakness about post-hoc strategy selection and the Figure 3 mislabeling. The consistency claim for CANON-Dynamic is weakened. Dropped as a standalone strength; kept only for CANON-Inter/CANON-Intra (Table 1).

---

## Novel Insights

CANON's most valuable conceptual contribution is the realization that DR.GRPO is a specific symmetric midpoint (μ=0.5) in a two-dimensional inter/intra advantage space, and that this "balanced" default may actively suppress useful signal: inter-group comparison sharpens exploitation of the metric direction that correlates with reward, while intra-group comparison promotes exploration within the disadvantaged group. The finding that these two roles have opposite effects on math vs. high-complexity logic (Figure 2, Table 1) — with complex tasks requiring exploration (CANON-Intra) and standard math tasks benefiting from exploitation (CANON-Inter) — is a genuine empirical insight that suggests the DR.GRPO default may systematically underperform at one end of the complexity spectrum regardless of any other design choices. This framing could inform future work on adaptive advantage estimation tied to task difficulty.

---

## Suggestions

1. **Fix Figure 3:** Either replace the embedded data table with actual values from Tables 1 and 2 scaled to a common axis, or clearly label Figure 3 as a conceptual schematic with illustrative values not drawn from experiments. The current version creates a false impression that the schematic values are experimental results, with the additional problem that "DR.GRPO" rows show CANON-Dynamic's actual numbers.

2. **Report CANON-Dynamic results for all four strategies side-by-side** (or at minimum the two best strategies) in the main text or a table, so readers can judge the variance across strategies. This makes the model-specific selection transparent rather than requiring readers to dig for it.

3. **Validate efficiency results with generation-time length limits** (not post-hoc truncation) for at least one α setting to confirm the budget-performance curves reflect actual model behavior under constrained generation.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| APA (advantage estimation for RLHF) | `/human_reviews/RtOTTdWbZd.md` | 5.25 (Reject) | Similar topic; less comprehensive evaluation, cleaner presentation, no data integrity issues |
| DeepSeek-Prover-V1.5 (RL for reasoning) | `/human_reviews/I4YAIwrsXa.md` | 6.25 (Accept Poster) | Stronger contribution with SOTA results; fewer methodology concerns |
| Low-score RL/LLM (in-context RL) | `/human_reviews/YW79lAHBUF.md` | 3.75 (Reject) | Much weaker; fundamental exploration issues |
| Low-score RL/LLM (R3HF token reward) | `/human_reviews/9LAqIWi3QG.md` | 3.0 (Reject) | Very weak; clearly worse than this paper |
| Medium-score RLHF (RLGF) | `/human_reviews/d98CzL5h0i.md` | 4.75 (borderline) | Similar range; comparable methodological quality |

**Reasoning:** This paper sits between APA (5.25, Reject) and DeepSeek-Prover-V1.5 (6.25, Accept). CANON's core contribution (inter/intra decomposition, DR.GRPO as special case) is more elegant than APA's, the multi-model evaluation is comprehensive, and the efficiency Pareto frontier (Section 5.3) is a credible additional contribution. However, the Figure 3 data integrity problem is a genuine concern — using CANON-Dynamic's actual experimental numbers in the "DR.GRPO" rows of Figure 3's table while showing schematic values for CANON methods is misleading and needs to be corrected. The CANON-Dynamic post-hoc strategy selection is an acknowledged methodological limitation that weakens the paper's headline cross-model claim. These issues collectively pull the score below DeepSeek-Prover-V1.5 (which had no comparable data presentation problems) and closer to APA. Given the solid core of Sections 4/5.1/5.3 but the meaningful concerns in Section 5.2 and Figure 3, the score sits at **5.5** (borderline reject).

**Axes:**
- *Originality*: Good — inter/intra decomposition is clean and novel
- *Importance of research question*: High — RLVR advantage estimation is a very active and practically relevant area
- *Claims well-supported*: Mixed — Table 1/3/4 data credibly supports CANON-Inter/Intra/Eff, but CANON-Dynamic's cross-model generalization claim is backed by post-hoc model-specific selection
- *Soundness of experiments*: Mixed — good breadth, but Figure 3 mislabeling and strategy selection methodology are real concerns
- *Clarity of writing*: Good
- *Value to community*: Meaningful, especially CANON-Inter/Eff contributions

**Score: 5.5**

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>