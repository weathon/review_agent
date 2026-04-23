Now I have enough information to write my final review. Let me synthesize everything.

## Summary

CoLR proposes a two-stage framework for temporal knowledge graph reasoning that constructs a Temporal Relation Structure Graph (TRSG) capturing cohesion between relations, uses it to guide path search via a Time-Fusion Search Graph (TFSG), and encodes extracted paths with a PLM and time sequence encoder. A Path Supplement Strategy (PSS) provides historical context for entity pairs lacking connected multi-hop paths. The paper also introduces three new benchmark datasets addressing accuracy (ACLED2023), inductive (ACLED-IND), and few-shot (ICEWS14-FS) evaluation gaps.

## Strengths

- **Principled TRSG formalization**: The cohesion-based Temporal Relation Structure Graph extends static RSG to temporal graphs via Theorem 1 (Eq. 2), providing a clean closed-form computation. Figure 4 validates that the TRSG captures stable structural patterns—diagonal lines reveal repetitive events, and cohesion distributions are strikingly similar across the three ICEWS datasets despite different relation counts, supporting the transferability claim.

- **New benchmark datasets addressing real gaps**: ACLED2023 uses 2023 events (post-PLM training cutoff) to mitigate information leakage; ACLED-IND uses geographic rather than random entity splits to preserve graph structural integrity; ICEWS14-FS creates a controlled few-shot setting. Each addresses a concrete, articulated deficiency in existing benchmarks (Section 6.1).

- **Effective handling of disconnected entity pairs**: The ablation (Table 3) confirms PSS addresses a real problem—removing it causes the largest performance drop (−12.49% MRR, −18.40% Hits@10), validating the paper's claim that prior methods suffer when no connected path exists between subject and object.

- **Comprehensive evaluation scope**: Seven datasets across transductive, inductive, and few-shot settings provide broad coverage. The cross-dataset transfer results (Table 2) show that CoLR's learned logical patterns generalize—e.g., trained on ICEWS14 and tested on ICEWS05-15 it achieves 78.69% MRR, exceeding the 76.82% when trained on ICEWS05-15 itself.

- **Modular two-stage design**: The framework cleanly separates path extraction (Stage 1) from encoding/scoring (Stage 2), and the authors note it is portable to enhance prior logical reasoning methods (Section 1, Appendix A.2).

## Weaknesses

### Fatal
None.

### Major

- **Inconsistent evaluation protocol for baselines on standard datasets**: The paper states "the experimental results of all baselines on the existing benchmark datasets were taken from prior papers" (Section 6.1), and only CENET was reproduced under a specified time-filtering setting. Different prior papers use different filtering protocols (static vs. time-aware) and dataset splits, which can shift MRR by 5–15+ points in TKG reasoning. Without verifying that all baselines use the same protocol, the 21–30% MRR improvements on ICEWS datasets cannot be reliably interpreted. The more modest improvements on YAGO (5.36%) and ACLED2023 (6.37%) are more credible since they involve fewer protocol ambiguities.

- **Baselines on new datasets use untuned ICEWS14 hyperparameters**: For ACLED2023, ACLED-IND, and ICEWS14-FS, the paper reports "we conducted experiments for each baseline using their parameter settings on ICEWS14" (Section 6.1). Using hyperparameters optimized for one dataset on structurally different datasets without any tuning systematically disadvantages baselines. Combined with CoLR's 16–18% MRR gaps on ICEWS14-FS and ACLED2023, this raises fairness concerns that a reviewer would weigh against acceptance.

- **Contribution attribution is incomplete—the PLM's isolated contribution is unknown**: The ablation (Table 3) shows that removing the paper's main structural contribution (TRSG) reduces MRR by only 2.97% (75.72 → 72.93), while PSS contributes 12.49% and TSE contributes 6.77%. However, there is no PLM-only control (e.g., encoding just the query (s, r_q, o) with ALBERT without any structural path input). The CoLR_{-RP} variant (relation-only paths) still achieves 71.29 MRR—17 points above the next-best baseline (54.01)—despite removing entity names from paths. This gap could largely reflect ALBERT's pre-trained semantic knowledge of relation text rather than learned structural reasoning. Without isolating the PLM's contribution, the paper cannot definitively attribute performance to its proposed structural components versus PLM pre-training.

### Minor

- **Framing mismatch: "coherent logical reasoning" vs. dominant PSS component**: The paper frames CoLR as performing multi-hop logical reasoning, but PSS—its largest performance contributor (12.49%)—provides single-edge historical context for disconnected entity pairs, not multi-hop logical paths. While the paper is transparent about PSS being a "supplement" (Section 5.1), the overall framing overemphasizes logical reasoning when the dominant mechanism is contextual enrichment. Understanding what fraction of test queries lack connected paths vs. use PSS would clarify the method's actual operating mode.

- **Table 2 labeled "Inductive results" but contains cross-dataset transfer**: The table header reads "Inductive results of CoLR on four datasets," but the text describes it as testing "cross-dataset application capabilities" (Section 6.2). Only the ACLED-IND column represents a genuine inductive setting (disjoint entity sets); the other columns show cross-dataset transfer (e.g., train ICEWS14, test ICEWS05-15). The label is misleading—these are different evaluation paradigms conflated under one heading.

- **Temporal scope of TRSG construction not explicitly specified**: The paper defines the ω time window for computing R_coh^ω (Section 4.2) but does not explicitly state whether the window slides over training-period subgraphs only or over the full TKG including test-period subgraphs during inference. Since the path search uses "historical subgraphs" (Section 5.1), the authors should clarify that TRSG weights are restricted to training data to rule out future information leakage.

### Trivial
None.

## Nice-to-Haves

- A PLM-only baseline (no TRSG, no PSS, no structured path search) that simply encodes (s, r_q) with ALBERT to score candidates. This would definitively separate structural contributions from PLM prior knowledge.
- Analysis of what fraction of test queries have connected multi-hop paths vs. rely on PSS. Given PSS contributes 12.49% MRR, this breakdown is essential for interpreting the method's behavior.
- Properly tuned baselines on new datasets, or at minimum a discussion of why ICEWS14 hyperparameters transfer reasonably.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"A 30% MRR improvement is not plausible under matched conditions"** (from Harsh Critic): This is speculation about what results *would* look like rather than a verified finding. The improvements on YAGO (5.36%) and ACLED2023 (6.37%) show the method works under more controlled conditions, but asserting that ICEWS improvements would collapse under matched conditions is unsubstantiated. The concern about inconsistent protocols is kept (Major), but the speculative conclusion is removed.

- **"PSS fabricates structural information"** (from Harsh Critic): The word "fabricates" implies dishonesty. PSS provides genuine historical context about entity neighborhoods—it's a single-edge contextual supplement, not fabricated data. The legitimate concern (kept as Minor) is that PSS is not "multi-hop logical reasoning," which is a framing issue, not fabrication.

- **"The max-over-paths scoring can inflate results especially when PSS provides fabricated paths"** (from Harsh Critic): Max-over-paths is a standard scoring strategy used in prior work (ALRE-IR, ILR-LR). The inflation concern is speculative without evidence that PSS-provided paths systematically bias toward correct answers.

- **"ACLED2023 only addresses fact memorization, not semantic knowledge leakage"** (from Harsh Critic): The paper explicitly acknowledges PLMs carry semantic knowledge (Section 6.1 and the ACLED2023 motivation). ACLED2023 addresses direct fact memorization, which is a concrete and testable concern. The semantic knowledge concern is a broader, harder-to-control issue that applies to any PLM-based method—it's a scope limitation, not a flaw in the ACLED2023 design.

- **"The path search uses P_next = P_time + P_coh additively with no learned weighting—this is a rigid heuristic"** (from Harsh Critic): Additive combination is a reasonable design choice; it's simple, interpretable, and works. Many successful methods use fixed combination weights. This is a design preference, not a weakness.

- **"No comparison of search efficiency (time, number of paths found) against alternatives"** (from Harsh Critic): Search efficiency analysis would be nice but is not required for the paper's claims. The paper focuses on prediction accuracy, not computational efficiency.

- **"Replace PLM with randomly-initialized transformer"** (from Harsh Critic): This is an interesting experiment but goes beyond what's needed. A PLM-only baseline (without structural components) would be more informative and directly addresses the attribution concern.

- **Generic strengths without specific citations** from Strength Finder: Dropped the generic "portable two-stage design" that didn't cite specific evidence beyond vague references; kept the substantiated version.

## Novel Insights

The ablation structure of CoLR reveals a pattern common to hybrid neural-symbolic methods: the most impactful component (PSS, 12.49%) operates as a fallback heuristic rather than the principled structural innovation (TRSG, 2.97%). This inversion between "primary contribution" and "primary performance driver" deserves attention in the community—it suggests that for sparse TKGs, the practical bottleneck is not *how to search better paths* but *what to do when no path exists*, and effective solutions may look more like contextual enrichment than multi-hop reasoning.

## Suggestions

- Re-run baselines on the three new datasets with at least minimal hyperparameter tuning (e.g., a small grid search), or clearly note that baseline results are untuned and provide an upper bound comparison.
- Add a PLM-only control condition: encode (s, r_q) text with ALBERT (no paths, no PSS, no TRSG) and score candidates via cosine similarity. This directly isolates the PLM's standalone contribution.
- Report the fraction of test queries that use PSS vs. have connected multi-hop paths, to clarify whether the method primarily performs logical reasoning or contextual enrichment.
- Relabel Table 2 as "Cross-dataset transfer and inductive results" to accurately represent the two different evaluation paradigms it contains.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| RoG (KG reasoning with relation paths + LLM) | ZGNWW7xZ6Q | 7.5 | Stronger than CoLR: cleaner baselines, better contribution attribution, interpretable reasoning with faithful grounding |
| INFER (Neural-symbolic TKG extrapolation) | ExHUtB2vnz | 5.5 | Comparable scope to CoLR (TKG reasoning, rule-based), similar limited novelty concerns, but fewer baseline fairness issues |
| TKG-LM (TKG + LM integration) | T0hhkuv8I0 | 4.0 | Similar PLM+TKG integration idea, weaker evaluation and fewer contributions than CoLR |
| RGMG (KG reasoning with RL) | d1zLRzhalF | 2.5 | Outdated baselines, limited novelty—CoLR has genuine structural innovations and new datasets that make it stronger |
| NPLL (Neural probabilistic logic for KG) | EGxgZzDODh | 3.0 | Overclaimed results without proper analysis, similar to CoLR's attribution gap but more severe (near-plagiarism concerns) |

CoLR sits between INFER (5.5) and TKG-LM (4.0). It has more substantial contributions than TKG-LM (TRSG formalization, three new datasets, comprehensive ablation) but shares INFER's concerns about limited novelty in its primary structural contribution and presentation issues. However, CoLR has a more serious baseline fairness problem than either, and its contribution attribution is less clear (TRSG = 2.97% vs PSS = 12.49% inverts the claimed importance). The paper is above the low-scoring anchors (RGMG at 2.5, NPLL at 3.0) due to genuine contributions in TRSG and new datasets, but below INFER (5.5) because the baseline fairness concerns are more damaging to the headline claims.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>