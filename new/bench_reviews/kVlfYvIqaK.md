Now let me search for calibration papers to score this appropriately.Now I have enough information to write the final consolidated review. Let me synthesize everything carefully.Now I have enough to write the final consolidated review.

---

## Summary
DyGPrompt (also referred to as DYGPROMPT in the paper) is a prompt learning framework for continuous-time dynamic graphs. It introduces *dual prompts* (a node prompt to bridge task-objective gaps and a time prompt to bridge temporal gaps between pre-training and downstream) and *dual condition-nets* (a time condition-net that generates time-conditioned node prompts, and a node condition-net that generates node-conditioned time prompts) to capture fine-grained evolving node-time patterns. The paper pre-trains on temporal link prediction and uses a similarity-based task template at downstream time, freezing backbone weights while tuning only the lightweight prompt/condition-net parameters. Experiments are conducted on four public datasets across node classification and transductive/inductive link prediction.

---

## Claims and Support

**Claim 1: DyGPrompt bridges both the task-objective gap and the temporal gap via dual prompts.**
*Partially supported.* Table 1 shows strong aggregate gains, but no controlled experiment separates the contribution of bridging task gaps (same task, shifted time) from bridging temporal gaps (different task, same time). The ablation in Table 2 conflates these two effects rather than isolating them.

**Claim 2: Dual condition-nets capture mutual node-time characterization better than fixed prompts.**
*Partially supported.* Variant 5 (node prompt + NCN) demonstrates a large jump in Wikipedia NC (+8.8 AUC) over the pure node-prompt baseline, confirming NCN's value. However, the stronger claim of *mutual characterization* is not rigorously tested; the experiments do not compare against one-sided conditioning only or parameter-matched alternatives to attribute the gain specifically to mutual interaction versus added capacity. The notation in Eq. (9) also reuses the same symbol on both sides, slightly obscuring the pipeline.

**Claim 3: DyGPrompt significantly outperforms state-of-the-art across node classification and link prediction in data-scarce settings.**
*Directionally supported, but scope must be qualified.* The strongest conventional DGNN baselines (TGAT, TGN) are placed in an unusual pre-train+limited-fine-tune regime that the paper explicitly acknowledges disadvantages them relative to their native use (Section 5.2 Remark). Within this few-shot adaptation regime, DyGPrompt's advantage is clear, especially on node classification and link prediction with TGN. The claim is reliable for the stated few-shot framing but should not be read as broad superiority over all dynamic graph methods.

**Claim 4: DyGPrompt is robust across DGNN backbones.**
*Mostly supported with notable exceptions.* Table 3 shows consistent gains on DyRep, JODIE, TGAT, and TGN. TREND and GraphMixer show smaller or occasionally negative gains (e.g., TREND-Reddit transductive: 80.42→79.62; JODIE-Reddit transductive: 59.81→58.89; GraphMixer-Reddit inductive: 57.64→57.43). The claim "regardless of backbone" overstates what the evidence supports.

---

## Strengths

- **Novel and well-motivated problem.** DyGPrompt is among the first frameworks to apply prompt learning to continuous-time dynamic graphs, addressing both task-objective and temporal gaps simultaneously. The dual-prompt design is a principled and non-trivial extension of static graph prompting.
- **Strong empirical results in Table 1.** DyGPrompt with TGAT achieves +12.88 AUC over TGAT-TiGPrompt on Wikipedia node classification, and TGN-DyGPrompt achieves substantial gains over TGN-TiGPrompt on link prediction across multiple datasets. Gains over the closest competitor (TiGPrompt) are large and consistent.
- **Parameter efficiency.** Freezing the backbone while tuning only the prompts and condition-nets makes the method lightweight and suitable for low-label regimes, which is the paper's stated scope.
- **Backbone generalizability.** Table 3 demonstrates improvement on six different DGNN backbones in the majority of configurations, showing the method is not heavily tied to one architectural choice.
- **Explicit acknowledgment of protocol differences.** The paper clearly notes that conventional DGNNs underperform their published results due to the pre-train+limited fine-tune regime (Section 5.2 Remark), which is honest.

---

## Weaknesses

### Fatal
*None. The paper's core empirical contribution is real and reproducible within its stated scope.*

### Major

- **Table 2 ablation is internally inconsistent and partially contradicts the text narrative.** As extracted from the paper, Variant 6 and DyGPrompt carry identical component checkmarks (node prompt ✓, time prompt ✓, NCN ✓, TCN ✓) yet yield different results (e.g., Wikipedia NC: 80.34 vs. 82.09). Additionally, the paper text says "Variant 2 (with node prompt) and Variant 3 (with time prompt) outperform Variant 1 (without these prompts)," but the table's checkmarks show Variant 1 carries the node prompt. Variant 1's results (67.00, 53.64, 59.27 on Wikipedia/Reddit/MOOC NC) exactly match the standalone TGAT numbers from Table 1, strongly suggesting Variant 1 is actually the no-prompt baseline, yet the checkmarks say otherwise. The paper cannot rest its component-level contribution claims on a table that is this difficult to interpret. The authors must correct the checkmark labeling, clarify what distinguishes Variant 6 from DyGPrompt, and make the ablation text consistent with the numbers.

- **Mechanistic claims about "why each component helps" are overclaimed.** The paper makes specific causal assertions: node prompts address task gaps, time prompts address temporal gaps, and dual condition-nets capture *mutual* characterization. None of these are validated by controlled experiments (e.g., same task / different time period, or ablations comparing one-sided vs. mutual conditioning). The text repeatedly asserts mechanism beyond what the ablation evidence supports. These claims should be scoped back to "consistent with the data" rather than stated as established.

- **Performance on the Genre dataset is near-random for all methods.** DyGPrompt achieves 52.03% AUC on Genre node classification vs. 51.46% for the next best method (ProG)—effectively indistinguishable from random. The paper does not discuss this nor acknowledge what it reveals about the method's limitations on this dataset. Counting Genre as a "win" inflates the claimed breadth of superiority.

### Minor

- **Eq. (9) notation is self-referential.** The update `p̃_{t,v}^{time} = p̃_{t,v}^{time} ⊙ f̃_t^{time}` reuses the same symbol on both sides. While the intended meaning (apply the time prompt to the output of the NCN) is inferable, it obscures the pipeline and should be corrected with a distinct symbol (e.g., `ĝ_{t,v}^{time}`).

- **Backbone robustness claim is slightly overstated.** "Regardless of backbone, DyGPrompt surpasses the original backbone...in almost all cases" (Section 5.4) is not fully accurate given TREND and JODIE exceptions. The language should be more conservative.

- **t-SNE visualizations (Figure 3) are expected by construction.** The prototype-based loss explicitly encourages class-based clustering, so Figure 3 confirms the loss works, not that the prompts have captured richer node-time semantics. This is a minor over-interpretation.

- **Negative interaction between dual prompts without condition-nets is unexplained.** In Table 2, Variant 4 (both prompts, no condition-nets, 72.25 Wikipedia NC) underperforms both Variant 2 (node prompt only, 72.59) and Variant 3 (time prompt only, 73.22), suggesting interference when prompts are combined without conditioning. The paper does not discuss this phenomenon, which is potentially informative.

### Trivial

- **Pre-training loss (Eq. 3) uses a single negative per positive** rather than a full InfoNCE denominator. This is less standard but follows cited prior work and is not likely to be a key source of performance difference in this setting.

---

## Nice-to-Haves

- Varying the amount of downstream training data (e.g., 10, 30, 100, 500 events) to validate when DyGPrompt's advantage is largest and whether standard fine-tuning eventually catches up—this would better characterize the method's practical regime.
- A controlled experiment separating task-gap bridging (same task as pre-training, shifted time window) from temporal-gap bridging (different task, same time window), to validate the proposed causal decomposition.
- Direct comparison of parameter counts and inference time against TiGPrompt and lightweight fine-tuning baselines to substantiate the "parameter efficiency" claim quantitatively.
- Discussion of the Genre dataset limitation—whether it is a dataset artifact, a class imbalance issue, or a genuine failure mode for the method.
- An iterative or alternating conditioning mechanism to further justify the "mutual characterization" framing (currently each condition-net conditions once, without feedback).

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **[Harsh Critic]** *"The ablation uses inconsistent variant definitions where Variant 1 has a node prompt but scores the same as the TGAT baseline, meaning the node prompt hurts performance."* — After careful reading, the most likely explanation is a PDF-extraction shift in the checkmark table (the text's ablation narrative is internally consistent with V1=no-prompt baseline). The Harsh Critic and Spark Reviewer both built their criticism on what appears to be a formatting artifact in the extracted text. The real problem is presentation/labeling ambiguity, not an inherent methodological flaw. **Retained as a Major weakness about table clarity**, but the framing that "node prompt alone hurts performance" is likely erroneous.

- **[Harsh/Spark]** *Full fine-tuning baseline on the same 30 events is missing.* — The paper's scope is explicitly few-shot prompt-based adaptation with a frozen backbone. A full fine-tuning comparison would be useful context but is outside the stated scope. Moved to Nice-to-Have.

- **[Spark]** *Simpler temporal adaptation baselines (per-timestamp learnable vectors, temporal shift vectors) are missing.* — Requesting these is scope creep into a different line of methods and would require significant additional experiments. Moved to Nice-to-Have.

- **[Human Finder / Harsh Critic]** *Missing related works (e.g., additional graph prompt references).* — Removed per the hard rule against missing related works.

- **[Harsh Critic]** *The evaluation protocol handicaps important baselines and therefore the paper's superiority claim is not valid.* — The paper explicitly acknowledges that TGAT/TGN are disadvantaged in their pre-train+limited fine-tune regime. This is inherent to the few-shot pre-training setup, not an unfair comparison design. The baselines are given favorable (asymmetric) treatment by being pre-trained on more data than the paper's method would require. Weakened to the Minor note that scope must be stated more clearly.

- **[Human Finder W6]** *Limited scope—no edge or graph-level tasks.* — The paper explicitly scopes to node classification and link prediction on continuous-time dynamic graphs, which is standard in the dynamic graph community. Removed as scope creep.

---

## Novel Insights

DyGPrompt's most empirically striking finding—largely underdiscussed by the paper itself—is that the **node condition-net (NCN) alone accounts for the vast majority of performance gains in node classification**. Variant 5 (node prompt + NCN only, no time prompt/TCN) achieves 81.40 on Wikipedia NC, very close to the full DyGPrompt (82.09). This suggests that time-conditioned node prompts—i.e., adapting node features to temporal context—is the primary driver of performance, while the node-conditioned time prompts (TCN) provide only marginal additional benefit. This has an important implication for future work: for node classification tasks, the temporal gap may be bridged primarily through time-aware node feature adaptation rather than through explicit time-prompt manipulation.

---

## Suggestions

1. **Fix and clarify Table 2.** Resolve the inconsistency between variant checkmarks and the text description. Clearly define what distinguishes Variant 6 from DyGPrompt (currently both appear as ✓✓✓✓). Use a blank row for the baseline (no prompt, equivalent to plain TGAT) and number variants consistently with the text.

2. **Scope the contribution claims.** Replace "DyGPrompt significantly outperforms various state-of-the-art baselines" with a claim tied to the few-shot pre-training adaptation regime. The current framing implies general superiority that the evaluation protocol does not support for conventional DGNNs.

3. **Address or explain the Genre results.** Either add a brief discussion of why Genre poses a near-random classification challenge (data sparsity, class imbalance, label noise) or explicitly exclude it from the "win" count. Near-random AUC for all methods suggests the task is ill-posed or the dataset unsuitable for this evaluation.

4. **Discuss the dual-prompt interference finding.** Variant 4 (both prompts, no condition-nets) underperforms each single-prompt variant on Wikipedia NC. This is a notable and informative result—explain why combining prompts without conditioning introduces interference, and use it to justify the design of condition-nets more strongly.

5. **Fix Eq. (9) notation.** Introduce a distinct intermediate variable for the NCN output before applying the time-prompt modulation, avoiding self-referential notation.

---

## Score and Decision

**Calibration anchors:**
- *yCN4yI6zhH* (GPromptShield – graph prompt learning, accepted poster): 6,6,6. A solid but incremental graph prompt paper with adequate experiments. DyGPrompt is comparably novel but has the Table 2 issue.
- *82Mc5ilInM* (FreeDyG – dynamic graph link prediction, accepted poster): 5,8,6,8. A clean, well-executed dynamic graph paper with rigorous evaluation. DyGPrompt has stronger novelty in problem framing but weaker ablation rigor.
- *OuxdVB6g1F* (TAGA – graph SSL, rejected): 3,6,5,5. Less coherent empirical story than DyGPrompt; DyGPrompt is clearly above this.
- *pIT0P1UASS* (Temporal Graph Scaling, rejected): 3,3,5,6. Much weaker empirical grounding than DyGPrompt.
- *4IT2pgc9v6* (One For All, spotlight): 10,6,6,6. A breakthrough paper; DyGPrompt is clearly below this.

DyGPrompt's empirical contribution is real and meaningful—it clearly extends prompt learning to dynamic graphs with strong results over TiGPrompt. The fatal weakness is the Table 2 presentation, which may reflect an extraction artifact but as presented makes the ablation story unreliable. The mechanism overclaiming and Genre issue are fixable. Given these issues, the paper is at the borderline of acceptability. I position it at **5.5**, slightly below the 6-level threshold used for clean work in the same space (yCN4yI6zhH), reflecting that the core contribution is publishable but requires non-trivial revision to the ablation section and claims before it meets acceptance standards.

**Originality:** Good — first dual-prompt framework for continuous-time dynamic graphs.
**Importance:** Good — the pre-train/downstream gap in dynamic graphs is a real and under-addressed problem.
**Claim support:** Fair — Table 1 numbers are strong; Table 2 has serious presentation problems undermining the mechanistic story.
**Experimental soundness:** Fair — strong primary benchmark but with acknowledged protocol asymmetries and an unreliable ablation table.
**Writing clarity:** Fair — the framework description is mostly clear but the ablation section is confusing.
**Community value:** Good — establishes a strong baseline for future prompt learning on dynamic graphs.

**Score: 5.5**
**Decision: Weak Reject** (revisions to the ablation table, scoped claims, and Genre discussion are necessary before acceptance)

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>