Now let me search for calibration anchors.Now I have enough to write the full review. Let me compile all my findings.---

## Summary
TABBY introduces a Mixture-of-Experts (MoE) post-training architecture modification to transformer-based LLMs for tabular data synthesis. By replacing standard MLP blocks or LM heads with column-dedicated expert sets (MMLP and MH variants), the paper claims higher-fidelity synthetic data generation. Experiments on six tabular datasets show that Plain-trained Tabby MH outperforms prior LLM-based tabular synthesis methods on 4/6 datasets in MLE, and matches real-data performance on the three classification benchmarks.

---

## Strengths

- **Strong empirical performance of Plain MH Tabby on classification datasets (Table 2):** Tabby MH under Plain training matches or exceeds the real-data MLE ceiling on Diabetes (74.3 ± 0.4 vs. 73.3), Travel (87.7 ± 1.2 vs. 87.5), and Adult (84.5 ± 0.2 vs. 84.5) — something no prior method achieved on all three simultaneously.

- **Robustness advantage on the Rainfall dataset (Table 2):** GReaT NT fails to generate any valid samples on Rainfall in any of three runs (N/A*), while Plain Tabby MH successfully generates data in all runs and achieves the best MLE of 0.58 among all methods — a qualitative failure mode the paper honestly documents with the asterisk convention.

- **Identification of a strong and overlooked baseline (Plain training, Section 4.0.1):** The paper shows that a simple, fixed-column-order LLM fine-tuning baseline with no shuffling substantially outperforms GReaT-family methods on several datasets. This finding directly challenges the field's reliance on more complex column-shuffling techniques and is a genuine, transferable insight.

- **Comprehensive empirical table (Table 2):** 15+ model and training-technique combinations across 6 datasets with variance over 3 runs, covering classification, regression, MLE, and discrimination metrics — more thorough than typical tabular synthesis papers.

- **Per-column loss monitoring as a training diagnostic (Figure 4, Section 4.3):** Because each column's loss is computed separately, practitioners can identify columns that are slow to learn (e.g., Median Income) or start very high but converge (e.g., Occupancy). This is a concrete, practical advantage over single-scalar-loss baselines.

---

## Weaknesses

### Fatal
None.

### Major

- **Parameter count confound undermines the central mechanistic claim (Tables 2 & 3):** Tabby MH expands the parameter count by a factor of V (the number of columns). Table 3 explicitly reports this: NT Distilled-GPT2 has 80M parameters while MH DGPT-2 has 270M (3.4×). No parameter-matched dense baseline is ever compared against. The paper attributes gains to the "dedicated expert per column" routing mechanism (Sections 3.1, 4.4), but this cannot be distinguished from a plain parameter-scaling effect without a controlled experiment. This is the most consequential methodological gap: if a 270M-parameter dense GPT-2 achieved similar results, the entire MoE routing contribution collapses to a scale story.

- **Expert routing under GReaT column-order shuffling is unexplained (Sections 3.1–3.3):** Section 3.1 states "the i-th column in the dataset is modeled by L_{a,i}." GReaT training (Section 4.0.1) randomizes the order ℓ_1, ℓ_2, …, ℓ_V at every step. The paper never clarifies whether expert selection is identity-based (column v_i always activates expert i regardless of its position in the sequence) or position-based (the j-th token position activates expert j). If position-based, different columns are processed by different experts each step, defeating the "dedicated expert per column" purpose. If identity-based, the mechanism for tracking which expert to activate for a given column in an arbitrary sequence is not described. This ambiguity is potentially load-bearing: it may explain why GReaT+Tabby consistently underperforms Plain+Tabby across Table 2, yet the paper provides no analysis or explanation of this pattern.

- **MMLP and MMLP-MH catastrophically fail on regression datasets (Table 2):** Plain MMLP achieves R² = 0.00 ± 0.00 on House (vs. Plain NT at 0.70 ± 0.11) and Plain MMLP-MH also achieves 0.00. This is not a marginal underperformance — these variants are completely non-functional on a major dataset category. The paper does not mention, let alone analyze, this failure. Since MMLP and MMLP-MH are two of the three Tabby variants tested, the paper's scope of applicability is implicitly restricted without being stated. Understanding this failure mode is important for establishing Tabby's actual use cases.

### Minor

- **Internal inconsistencies in the conclusion:** The conclusion (Section 5) states "Tabby reaches parity with non-synthetic data in two out of three evaluated datasets, according to machine learning efficacy with a Decision Tree Classifier." Multiple errors: (a) the evaluation used a Random Forest (Section 4.0.3), not a Decision Tree Classifier; (b) "two out of three evaluated datasets" conflicts with six datasets being evaluated in Section 4.1 and parity on three classification datasets claimed in the abstract and Section 4.1. These are straightforward internal inconsistencies that erode reader confidence.

- **Weak evidence base for Claim 2 (Section 4.2):** The model-size scaling experiment uses a single dataset (a 6-column, 5160-row subset of House), only 5 training epochs (vs. 50 in Section 4.1), and a single pair of model comparisons (DGPT-2 vs. Llama). The conclusion that "Tabby allows smaller LLMs to achieve fidelity more similar to that of LLMs with higher parameter counts" is drawn from this narrow evidence. This claim would need at least the same six-dataset scope as Claim 1 to be credible as a general architectural property, and ideally would compare Tabby DGPT-2 to a parameter-matched dense model.

### Trivial

- The description of the routing mechanism uses the term "Gated Mixture-of-Experts" in the title/citations but implements a fully deterministic, hard routing (column index → expert). There is no learned gate. This minor terminological mismatch between the Shazeer (2017) citation and the actual mechanism could be clarified with one sentence.

---

## Nice-to-Haves

- A parameter-matched ablation (e.g., full GPT-2 Medium or Large at ~270M parameters, as a dense baseline) would definitively establish whether improvements come from the routing mechanism or from scale.
- An ablation comparing position-based vs. identity-based routing under GReaT would clarify whether the mechanism functions as intended under column shuffling, and would explain the consistent GReaT+Tabby vs. Plain+Tabby gap.
- Expanding the Claim 2 experiment to all six datasets with the same 50-epoch budget would substantially strengthen the architectural scaling claim.
- Analysis of why MMLP fails on regression: are numerical columns particularly poorly served by MoE MLPs? This would clarify the scope of each Tabby variant.

---

## Removed Points

*These points are flagged to be removed — treat them with caution.*

- **Harsh Critic: "Gated MoE" is misleading terminology (Section 3.1).** The paper cites Shazeer et al. (2017) for MoE but uses deterministic routing. This was classified as a trivial terminological imprecision rather than a substantive structural error; the mechanism is described correctly in the equations regardless of the label.

- **Harsh Critic: EOC token initialization not specified (Sections 3.2–3.3).** The paper introduces `<EOC>` without detailing tokenizer vocabulary changes or initialization. Under the hard rule on trivial implementation details, this is not a substantive weakness — it is a minor reproducibility note about a small engineering decision.

- **Harsh Critic: Claim 3 (per-column loss tracking) "inflates apparent scope."** While the criticism that observational training byproducts do not constitute a research contribution has merit, the per-column loss diagnostic is presented honestly and modestly. It is retained as a minor strength but removed from the "weakness" roster.

- **Strength Finder: "Tabby enables smaller models to punch above their weight class" (Table 3/Figure 3).** This strength directly conflicts with the major weakness about the parameter count confound. A 270M-parameter model outperforming an 80M-parameter model is not architecturally surprising. Since the weakness wins when strength and weakness conflict, this strength is removed from the main review.

- **Harsh Critic: "Plain NT turns out to be quite competitive — contributions are entangled."** While the entanglement point is partially valid, the paper's value-add is clearly the Tabby MH architecture, and Plain MH consistently beats Plain NT. The contribution is not fully disentangled from Plain training gains but this is already covered under the parameter confound weakness.

---

## Novel Insights

The most genuinely underappreciated finding in the paper is not the MoE routing itself but the Plain training baseline result: a plain fixed-order LLM fine-tuning with no shuffling beats GReaT-family methods (which use column-order randomization, pretraining on tabular corpora, and categorical encoding tricks) on most datasets. This suggests that the column-shuffling technique central to GReaT may introduce distributional artifacts that hurt more than help, and that the community has been over-engineering the training pipeline relative to what a simpler approach can already achieve. If this finding holds broadly, it constitutes an important correction to prior LLM-based tabular synthesis literature.

---

## Suggestions

1. Add a 270M-parameter dense GPT-2 baseline (GPT-2 Medium/Large) evaluated on the same six datasets. If Tabby MH DGPT-2 (270M) outperforms it, the routing mechanism gets credit; if not, the paper's story needs to be reframed around scale.
2. Explicitly state and ablate the routing policy (position-based vs. identity-based) under GReaT shuffling, and explain the consistent performance gap between GReaT+Tabby and Plain+Tabby.
3. Fix the conclusion: replace "Decision Tree Classifier" with "Random Forest," and correct the dataset count ("three out of six" or "three out of three classification datasets").
4. Provide a brief diagnostic analysis of why MMLP fails on the House regression dataset, and scope the recommendations accordingly.

---

## Score and Decision

**Calibration anchors:**

| Path | Avg Score | Decision | Relevance to TABBY |
|---|---|---|---|
| 4Ay23yeuz0 (TabSyn) | 6.75 | Accept (Oral) | Strongest tabular synthesis anchor; clear methodology, no parameter confound, strong results |
| QPtoBPn4lZ (CDTD) | 5.50 | Accept (Poster) | Tabular diffusion synthesis; accepted despite incremental novelty concerns |
| wT1aFmsXOc (tabular diffusion memorization) | 5.00 | Reject | Tabular synthesis with methodology gaps |
| hz2zhaZPXm (TabFMs) | 3.50 | Reject | LLM fine-tuning for tabular data; unclear results, rejected |
| 3qDhqj6qfu (TabKANet) | 3.00 | Withdrawn/Reject | Tabular data modeling with KAN, limited novelty |
| zB6uMznFuZ (TimeAutoDiff) | 3.00 | Withdrawn/Reject | Tabular synthesis with weak methodology |
| kzePnQWUvC (tabular data distillation) | 3.33 | Withdrawn/Reject | Tabular synthesis with overclaims and gaps |

**Positioning:** TABBY is clearly above the 3.0–3.5 cluster: it introduces a genuinely novel architectural idea (first LLM arch modification for tabular synthesis), provides comprehensive experiments, and identifies a practically important finding about Plain training. However, it falls short of the CDTD (5.5) level because: (i) the central mechanistic claim is undermined by an uncontrolled parameter count confound; (ii) two of three Tabby variants catastrophically fail on regression without explanation; (iii) the conclusion contains factual errors; and (iv) the routing mechanism is ambiguous under the primary GReaT training condition. These gaps are real but not fatal — the best Tabby variant (Plain MH) does show strong results, and with a parameter-matched ablation the contribution could be substantially established. A score of **4.5** places it between the rejected tabular papers and the borderline-to-weak accepted papers like CDTD.

**Originality:** Moderate — first LLM architecture modification for tabular synthesis, but the mechanistic case for why MoE routing specifically helps is unestablished.  
**Importance of research question:** High — tabular synthesis is underserved relative to text/image.  
**Claims vs. support:** Partially supported — headline results hold for one variant (MH) under one training method (Plain), but confounded by scale; regression claims are undermined by MMLP failure.  
**Soundness of experiments:** Moderate — broad but inconsistently reported, missing critical parameter ablation.  
**Clarity of writing:** Fair — the main body is readable but the conclusion has factual errors that undermine credibility.  
**Value to community:** Moderate — the Plain training insight and the failure analysis of GReaT on Rainfall are genuinely useful; the MoE contribution needs better validation.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>