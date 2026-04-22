The paper does acknowledge that FormalAlign is "our fine-tuned Mistral-7B model" and that GPT-4/GPT-3.5 use "query prompts" (Appendix C.2). However, the paper doesn't frame the comparison as fine-tuned vs. zero-shot explicitly and doesn't discuss this as a confound.

Now I'll write the final review.</think>## Summary

FormalAlign proposes the first automated framework for evaluating semantic alignment between informal mathematical statements and their formal (Lean 4) counterparts in autoformalization. The method jointly trains a model (Mistral-7B) on the autoformalization sequence generation task (cross-entropy loss) and a representational alignment task (contrastive loss), producing an alignment score combining token-level certainty and embedding-level similarity. Evaluation is conducted on four benchmarks augmented with six synthetic misalignment strategies.

## Strengths

- **Excellent problem identification and motivation**: Figure 1 concretely demonstrates that compiler validity and BLEU are insufficient for alignment evaluation — a misaligned formalization (`80*c=63` instead of `63*a=80*c`) passes both checks. This identifies a genuine and important gap in the literature.

- **Substantial precision improvement over GPT-4 on in-domain data**: Table 3 shows FormalAlign achieves 93.65% precision on FormL4-Basic vs. GPT-4's 42.68%, demonstrating strong reduction in false positives for alignment detection.

- **Contrastive loss provides meaningful OOD generalization gains**: Table 5 shows adding contrastive loss to CE improves MiniF2F-Valid AS from 52.45% to 66.39% (a 14-point gain), confirming the dual-loss design helps on the harder out-of-domain benchmarks.

- **Similarity score complements certainty score on OOD data**: Table 6 shows that combining the similarity score with the certainty score lifts MiniF2F-Valid from 53.69% to 66.39% (12.7-point gain) and MiniF2F-Test from 55.55% to 66.70% (11.15-point gain), validating the V_align inference design for generalization.

- **Framework generalizes across model architectures**: Table 4 demonstrates applicability across Phi2-2.7B, DeepSeekMath-7B, LLaMA2-7B, and Mistral-7B, with consistently improving with model scale.

- **Efficiency advantage over human evaluation**: Section 4.4 reports human experts require ~3 hours for 80 items vs. FormalAlign's <2 minutes, with 65% vs. 79.58% correctness — a significant speed-correctness tradeoff.

## Weaknesses

### Fatal

- **Numerical inconsistencies between abstract/introduction and Table 3**: The abstract claims GPT-4 achieves 88.91% AS on FormL4-Basic, but Table 3 reports 90.23%. The introduction claims GPT-4 precision on FormL4-Basic is 26.33%, but Table 3 shows 42.68%. The introduction claims GPT-4 MiniF2F-Valid AS is 64.34%, but Table 3 shows 67.24%. The "11.58% higher" claim in the abstract (99.21−88.91=10.30, not 11.58) is also arithmetically wrong. Multiple numbers in the abstract and introduction do not match the primary results table. This raises serious concerns about the reliability of the reported results and whether the abstract reflects the actual experiments.

### Major

- **Entire evaluation uses synthetic misalignments, never real autoformalization model outputs**: The paper's central motivation (Section 1, Figure 1) is detecting semantic misalignments that autoformalization models actually produce. Yet the evaluation (Table 3, Section 4.3) exclusively uses synthetic perturbations (constant/exponent modification, unused variable injection, type change, equality swap, random pairing). These are designed to be detectable — they produce formal statements with obvious structural differences from ground truth. There is no evaluation on outputs from any actual autoformalization system, so there is no evidence FormalAlign works on its intended task. The motivating Figure 1 error (variable role confusion in equations) is precisely the kind the synthetic strategies fail to capture.

- **Dominance of trivial negatives in in-domain evaluation inflates headline results**: On FormL4-Basic, 86.42% of negatives come from "variable_new" (46.42%, adding an unused variable declaration) and "random pairing" (40.00%, pairing with a formal statement from a completely different problem) — both trivially distinguishable. The 93.65% precision and 99.21% AS on FormL4-Basic are thus much less informative about real alignment checking than they appear.

- **Recall on OOD data is severely limited**: FormalAlign achieves only 60.66% and 63.37% recall on MiniF2F-Valid and MiniF2F-Test, compared to GPT-4's 89.87% and 92.88% (Table 3). This means ~37% of true alignments are missed on the more realistic benchmarks. For a system claiming to "significantly reduce the need for manual verification" (abstract), a 37% false negative rate means many misaligned pairs would pass undetected — undermining the core practical claim. No F1 scores, precision-recall curves, or sensitivity analysis for the θ=0.7 threshold are provided to contextualize this tradeoff.

- **Headline comparison with GPT-4 conflates fine-tuning with method contribution**: The abstract and introduction prominently claim FormalAlign "outperforms GPT-4" without acknowledging that this comparison is between a fine-tuned Mistral-7B and a prompted GPT-4. A fine-tuned 7B model outperforming a prompted frontier model on a narrow targeted task is expected, not surprising. This framing inflates the perceived contribution.

### Minor

- **Overclaimed "consistent performance across four diverse datasets"**: AS drops from 99.21% (FormL4-Basic) to 66.39% (MiniF2F-Valid) — a 33-point decrease. The "mutually enhancing" description is also overstated given that contrastive loss alone is near-useless (Table 5: 36–59% AS) and its contribution in-domain is negligible (FormL4-Basic: 98.64→99.21, +0.57 points).

- **No justification or sensitivity analysis for θ=0.7**: Section 4.2 sets the alignment detection threshold at θ=0.7 without justification. Given the 21:1 negative-to-positive ratio and the dramatically different precision-recall tradeoffs across datasets, a sensitivity analysis would be informative.

### Trivial

- None.

## Nice-to-Haves

- Evaluation on real autoformalization model outputs (e.g., GPT-4, ByT5, or LLM-based autoformalizers) would directly validate the method's practical utility.
- Harder synthetic misalignment strategies (e.g., variable role confusion as in Figure 1, omitted hypotheses, swapped logical connectives) would make the evaluation more realistic.
- Fine-tuned GPT-3.5 as a baseline (same training data, without contrastive loss) would isolate the method's contribution from the benefit of fine-tuning itself.
- F1 scores and precision-recall curves across thresholds would provide a more complete picture of the precision-recall tradeoff.

## Removed Points

These points are flagged to be removed, treat them with caution:

- *Harsh critic claim that "the similarity score contributes almost nothing"*: This is only true on in-domain data (FormL4: 98.98→99.21). On OOD MiniF2F, the similarity score provides a critical 12–13 point boost (Table 6: 53.69→66.39), so the component does contribute meaningfully where it matters most. The harsh critic ignored the OOD ablation results.

- *Harsh critic claim about "in-batch negatives being correctly aligned formal outputs" as a training deficiency*: This is how standard contrastive learning works — negatives are other examples in the batch. The model learns to distinguish *this* formal statement from *other* formal statements for a given informal input. This is a reasonable proximate objective for alignment detection; the critic is confusing the training objective with the downstream task.

- *Strength finder's claim about "competitive with GPT-4 on OOD benchmarks"* (66.39% vs. 67.24%): This is misleading — while AS is comparable, FormalAlign's recall is vastly inferior (60.66% vs. 89.87%), and on MiniF2F-Test, FormalAlign actually underperforms GPT-4 (64.61% vs. 70.82%). This "strength" obscures the real recall problem.

- *Strength finder's claim about "substantially higher precision" as a key practical benefit*: While precision is higher, the 37% recall failure rate means the system misses many aligned pairs, which is a serious practical deficiency for a verification tool.

- *Missing experiments/proofs in appendix*: Removed per rules — the parser strips appendix content from all papers.

- *Formatting/typo-related complaints*: Removed per rules.

## Novel Insights

The paper introduces a valid and under-explored evaluation problem (automated alignment checking for autoformalization), but the evaluation methodology reveals a fundamental circularity: the synthetic misalignment strategies are designed to be detectable by surface-level or structural differences, yet the method is supposed to detect deep semantic misalignment. The gap between the motivating example (Figure 1) and the evaluation design is the paper's core limitation — the system is validated on what it can easily detect rather than on what it needs to detect. The substantial recall degradation on OOD data further suggests that the model's alignment signal is largely derived from token-level generation certainty (a well-known proxy) rather than from genuine semantic understanding of the informal-formal correspondence.

## Suggestions

- **Most critical**: Run FormalAlign on outputs from actual autoformalization systems (even 50–100 examples with manual annotation) to validate whether the method detects real-world misalignments.
- Report F1 scores and precision-recall curves; the current presentation hides the precision-recall tradeoff.
- Fix the numerical inconsistencies between the abstract, introduction, and Table 3 — these undermine trust in the results.
- Acknowledge the fine-tuned vs. zero-shot comparison explicitly and add a fine-tuned baseline (e.g., GPT-3.5 fine-tuned on the same data without contrastive loss).
- Reduce or eliminate "random pairing" and "variable_new" from FormL4 evaluation, or at minimum report results stratified by misalignment difficulty.

## Calibration Anchors

| Paper | Avg Score | Comparison |
|-------|-----------|------------|
| BEq (hUb2At2DsQ) — automated evaluation for autoformalization | 7.20 | More rigorous: evaluates on expert-annotated real equivalence pairs (200 pairs), proposes neuro-symbolic method, adds OOD benchmark. FormalAlign is weaker in evaluation design (synthetic only, no real-world validation) and has numerical inconsistencies, but targets a complementary evaluation angle. |
| SynBench (9RLC0J2N9n) — synthetic evaluation benchmark | 4.50 | Similar weakness: synthetic proxy evaluation with questionable real-world validity. FormalAlign has the same core issue but additionally suffers from numerical inconsistencies and overclaimed GPT-4 comparison. |
| Fine-tuned vs zero-shot unfair comparison papers (pE0UM18TQh, RUn41kd6i0) | 4.00–4.33 | Similar overclaim pattern. These were rejected. FormalAlign's comparison is less egregious (it doesn't hide the fine-tuning) but still inflates claims in the abstract. |
| Numerical inconsistency papers (CahIEKCu5Q, 9fvnZRCGra) | 5.00–5.50 | Multiple number mismatches between abstract and tables. FormalAlign has a similar pattern but more pervasive (3+ inconsistencies across key tables). |
| Synthetic Lean theorem generator (EeDSMy5Ruj) | 5.00 | Similar domain (Lean/formal math), limited real-world validation, modest experimental gains. Slightly weaker than FormalAlign in problem novelty but had cleaner reporting. |
| miniCTX (KIgaAqEFHW) — formal theorem proving benchmark | 8.00 | High bar: novel benchmark with rigorous evaluation. FormalAlign falls well below this standard due to synthetic-only evaluation. |

**Score reasoning**: The paper identifies an important problem and proposes a reasonable framework, but has three compounding major issues: (1) the evaluation never tests on real autoformalization outputs, (2) there are significant numerical inconsistencies between the abstract and the main results table, and (3) the fine-tuned vs. prompted GPT-4 comparison is presented without acknowledging the asymmetry. These place it below the BEq paper (7.2) which solves a similar problem with real-world validation, and below the numerical-inconsistency borderline papers that scored 5.0–5.5 but had fewer reporting errors. The synthetic-only evaluation without real-world validation aligns with papers scoring 4.5–5.0. However, the problem identification and framework design are genuinely valuable. I position this at **5.0** — borderline, with the numerical inconsistencies being the most concerning issue that could be resolved in revision, but the synthetic-only evaluation is a fundamental limitation.

## Score and Decision

Score: 5

Decision: Borderline — the problem is important and the framework design is reasonable, but the evaluation is never conducted on real autoformalization model outputs, headline numbers are inflated by easy synthetic negatives and a fine-tuned vs. zero-shot comparison, and there are significant numerical inconsistencies between the abstract/introduction and Table 3. These issues collectively undermine confidence in the core claims.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>