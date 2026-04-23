Now let me run calibration searches in parallel:Let me read one anchor paper more carefully for calibration:Now I have enough information to write the comprehensive final review.

---

## Summary

UNSTAR proposes using "anti-samples"—paraphrased questions paired with incorrect answers and misleading rationales—to induce selective forgetting in LLMs. The paper claims three contributions: (1) anti-sample-induced unlearning as a novel paradigm, (2) misleading rationales (STaR-style justifications) to accelerate forgetting, and (3) fine-grained targeted unlearning that selectively erases specific associations (e.g., Harry Potter → Hogwarts) while preserving all related knowledge. Evaluation on WPU against eight baselines claims UNSTAR achieves perfect (100/100) scores on both Unlearning Efficacy and Model Utility simultaneously.

---

## Strengths

- **Fine-grained targeted unlearning framing (Section 3, Table 4)**: The distinction between erasing *all* knowledge about an entity versus erasing only a specific association between two entities is conceptually valuable and underexplored. Table 4 shows concretely that standard targeted unlearning corrupts collateral facts ("Harry Potter is a British actor, writer, and director...") while UNSTAR's approach preserves them.

- **Comprehensive multi-dimensional evaluation framework (Section 4.1)**: The five composite metrics—Unlearning Efficacy, Model Utility, Response Quality, Hallucination Avoidance, and Adversarial Robustness—are more thorough than the ROUGE-only evaluations common in prior work, and the inclusion of jailbreak-based Adversarial Robustness is an important, often-neglected dimension.

- **Eight-baseline comparison on WPU (Figure 2)**: Evaluating against GA, NPO, Prompt, Prompt-distill, DI, WHP, WHP+, and RWHP provides a broad comparative landscape for the WPU dataset.

---

## Weaknesses

### Fatal

*(None that outright invalidate the core conceptual contribution, but the issues below collectively undermine confidence in the results.)*

### Major

- **Verified text-table contradiction in Section 4.2**: The narrative text describing Figure 2 reports fundamentally different numbers from the actual table. Specifically:
  - *Response Quality*: Text says "UNSTAR scores slightly lower here (92)" and "GA (0) and NPO (24) perform poorly," but the table shows UNSTAR = 100, GA = 100, NPO = 80.
  - *Model Utility*: Text says "GA (13) and WHP (93)" but table shows GA = 10 and WHP = 100.
  - *Hallucination Avoidance*: Text says "UNSTAR (83), Prompt-distill (98), RWHP (86)" but table shows UNSTAR = 100, Prompt-distill = 100, RWHP = 85.
  
  These are not rounding errors — they are large-magnitude discrepancies that indicate the prose was written from a different version of the experiments than the final table. This raises serious concerns about which set of numbers reflects actual experimental outcomes.

- **Suspicious perfection in Figure 3**: The reported data for Iterations vs. Unlearning Efficacy is: {0:10%, 10:15%, 20:25%, 30:35%, 40:45%, 50:55%, 60:65%, 70:75%, 80:85%, 90:95%, 100:100%}. From iteration 20 onwards this is an exactly uniform Δ10 per 10 iterations with zero variance. Real averaged experimental measurements across 100 entities and multiple paraphrase seeds cannot produce this pattern. This data appears constructed rather than measured.

- **Two of three advertised datasets produce no comparative results**: Table 1 presents statistics for WPU, Peter Parker, and TOFU; Table 2 provides hyperparameters for all three. Yet Figure 2 — the sole comparative evaluation — covers only WPU. No results for Peter Parker or TOFU against any of the eight baselines are reported anywhere. Given that TOFU is a standard LLM unlearning benchmark (Maini et al., 2024) with established baselines, its omission is a significant gap in the paper's empirical support.

- **Central contribution (misleading rationales) is never ablated**: Contribution ❷ explicitly claims that misleading rationales drive UNSTAR's performance. However, there is no experiment comparing UNSTAR (with rationale generation) against the obvious baseline of fine-tuning on (question, wrong answer) pairs *without* rationales. Without this ablation, it is impossible to determine whether the gain comes from the STaR-style justification mechanism or simply from iterative fine-tuning on paraphrased incorrect answers — a much simpler approach already partially explored by prior methods.

- **Fine-grained targeted unlearning (Contribution ❸) supported only by cherry-picked qualitative outputs**: Table 4 presents nine qualitative responses for a single fictional entity (Harry Potter/Hogwarts). No metric is defined, no success rate is computed, and no evaluation across the 100 or 200 forget targets in WPU/TOFU is reported. Claiming a capability "not achievable by previous works" requires quantitative evidence across a meaningful sample.

### Minor

- **Algorithm 1 termination condition differs from prose description**: Section 3 (step 3b) states the unlearning check is "if â ≠ a" (model answer ≠ correct answer). Algorithm 1 line 4.2 states the condition as "â ≠ â_bar" (model answer ≠ the specific incorrect target answer). These are logically distinct: a model could output any wrong answer satisfying â ≠ a without outputting the specific â_bar. The algorithm as written has a different semantics from the prose, and neither version is fully defended.

- **UNSTAR claiming 100/100 on Unlearning Efficacy and Model Utility simultaneously**: Achieving perfect forgetting while perfectly preserving all retain-set knowledge is the fundamental challenge of unlearning and the precise trade-off that prior methods struggle with. The paper offers no explanation of why UNSTAR avoids this trade-off, nor any analysis of whether the evaluation metrics are sensitive enough to detect subtle degradation. Given the text-table inconsistency noted above, these scores require further scrutiny.

- **Normalization prevents absolute comparison**: All scores in Figure 2 are normalized by the maximum across all methods. Since UNSTAR achieves the maximum on four of five metrics, all other methods are scored relative to UNSTAR's ceiling. Raw metric values (ROUGE-L scores, GPT privacy/quality scores, rejection rates) are never reported, making it impossible to assess whether score differences are practically meaningful.

### Trivial

- The RL policy gradient framing (Equations 1–2) provides limited insight beyond the algorithmic description: the paper immediately acknowledges that UNSTAR approximates J by "greedily decoding" and "taking multiple gradient steps on the same batch," which is supervised fine-tuning on curated samples. The formalism is not incorrect but adds little explanatory power.

---

## Nice-to-Haves

- Include membership inference attacks or probing studies to distinguish genuine unlearning (removal of internal representations) from answer-override (model still encodes the fact but outputs a different answer).
- Report wall-clock training times; the Apple M3 Pro 18 GB setup makes computational efficiency claims ("anti-samples offer an efficient…strategy") harder to assess without absolute runtime data.
- Define a quantitative metric for fine-grained targeted unlearning (e.g., fraction of target association successfully erased vs. fraction of collateral related facts altered) and measure it across all forget-set entities.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **Harsh Critic Issue: "RL framing is just supervised fine-tuning"**: The paper itself acknowledges that UNSTAR approximates the RL objective via greedy decoding and multiple gradient steps. Criticizing this approximation as "mere supervised fine-tuning" conflates presentation choice with methodological error. Removed as overstated.

- **Harsh Critic: Anti-sample concept "understates prior work"**: WHP and Jang et al. already use wrong/alternative answers. The harsh critic argues this makes UNSTAR's framing unoriginal. However, the paper explicitly distinguishes UNSTAR from these via (a) paraphrase diversity, (b) rationale generation, and (c) fine-grained association-level targeting. The related work section acknowledges these prior methods. Removed as insufficiently specific to constitute a remaining weakness.

- **Strength Finder: "RL policy gradient formulation as supporting strength"**: Dropped — it is a presentation choice that the paper itself qualifies as an approximation, not a standalone methodological contribution.

- **Strength Finder: "Iterative convergence evidence (Figure 3)"**: Dropped — Figure 3 data is under verified suspicion of being fabricated/schematic (perfectly linear, zero variance). Cannot be retained as a strength.

- **Strength Finder: "Reproducible experimental setup using same settings as RWHP"**: Dropped as generic — this is a baseline requirement, not a strength.

---

## Novel Insights

The paper's most conceptually valuable observation — that prior targeted unlearning conflates entity-level and association-level forgetting, causing collateral damage to related facts — is a real and underarticulated gap in the field. The anti-sample paradigm (fine-tuning on wrong-answer-with-justification tuples) is a distinct and interesting methodological lens even if it has partial predecessors. If the experimental integrity concerns (text-table contradictions, Figure 3, missing dataset evaluations) are resolved and the rationale ablation is run, there is a publishable contribution here. The integrity issues, however, are currently serious enough to prevent acceptance.

---

## Suggestions

1. **Reconcile text and table in Section 4.2**: Verify which set of numbers (text narrative or Figure 2 table) reflects actual experimental results and rewrite accordingly. This is the most urgent fix.
2. **Add the rationale ablation**: Run UNSTAR without the justification generation step (just fine-tune on wrong answers) and include in Figure 2. This directly validates Contribution ❷.
3. **Report Peter Parker and TOFU results**: Add a comparative table for these datasets, or honestly scope the paper to WPU and remove the other datasets from Table 1 and 2 if results are not ready.
4. **Define and measure fine-grained unlearning quantitatively**: Create a metric separating target association erasure from collateral fact preservation; report it across all forget-set entities.
5. **Replace or justify Figure 3**: If the data is real, explain why it is perfectly linear; if it is schematic, label it as such. Report actual experimental variance.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Human Score | Comparison to UNSTAR |
|---|---|---|
| `/human_reviews/MGKDBuyv4p.md` | 7.33 (Spotlight) | Strong benchmark study with multiple memorization methods, clean evaluation, no data integrity concerns — clearly stronger than UNSTAR. |
| `/human_reviews/PDct7vrcvT.md` | 5.25 (Withdrawn) | LLM unlearning with WHP on single dataset (Harry Potter only), evaluated without baselines, but data was self-consistent; weaker empirically but no integrity issues. |
| `/human_reviews/dXCpPgjTtd.md` | 6.0 (Poster) | Large-scale knowledge washing paper with consistent results and ablations — more complete evaluation than UNSTAR. |
| `/human_reviews/1ExfUpmIW4.md` | 6.0 (Poster) | LoKU unlearning with consistent experiments and ablation studies — methodologically more rigorous. |
| `/human_reviews/TJU9J8iQXL.md` | 2.33 (Withdrawn) | Data inconsistencies across experiments, flawed fairness metrics — similar integrity class to UNSTAR; this is the closest qualitative match. |
| `/human_reviews/WFlLqUmb9v.md` | 2.5 (Reject) | Missing ablations against simpler alternatives, overclaimed results — directly parallels UNSTAR's missing rationale ablation and claimed perfect scores. |
| `/human_reviews/3ZdGSTxKuy.md` | 2.0 (Withdrawn) | Claims too large for evidence, unfair comparisons — slightly weaker framing than UNSTAR. |

UNSTAR's conceptual contribution (fine-grained targeted unlearning, anti-sample framing) is genuine and more novel than most papers in the low-score cluster. However, the text-table contradiction in the main results (verified), the suspicious linearity of Figure 3 (verified), the missing evaluation for 2/3 advertised datasets (verified), and the unablated central contribution place it firmly below the 6.0-level accepted unlearning papers. The data integrity concerns align most closely with TJU9J8iQXL (2.33) and WFlLqUmb9v (2.5), which had similar patterns of inconsistent results and missing ablations. The paper's stronger conceptual framing warrants a slight upward adjustment from those anchors.

**Final Score: 3.0 / Reject**

MY FINAL SCORE: <pineapple>3.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>