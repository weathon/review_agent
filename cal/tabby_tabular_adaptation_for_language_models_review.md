=== CALIBRATION EXAMPLE 42 ===

# Final Consolidated Review
## Summary
Tabby is a post-training architecture modification for transformer-based LLMs that replaces designated MLP or LM-head blocks with V column-specific expert copies (one per dataset column), allowing each column to be modeled by a dedicated set of parameters. Applied to Distilled-GPT2, the MoE LM Head (MH) variant achieves competitive tabular synthesis quality on 6 benchmark datasets. The paper additionally identifies that a simple "Plain" training technique—previously absent from LLM-based tabular synthesis literature—provides a strong baseline.

---

## Strengths

- **First architecture-level modification for LLM tabular synthesis.** Prior work (GReaT, TapTap, Tabula) exclusively contributes training techniques. Tabby changes the model itself, and this distinction is well-argued and clearly motivated by the semantic independence of tabular columns.

- **Plain training is a non-trivial insight.** The discovery that plain sequential column training, absent from all prior LLM tabular synthesis works, achieves near-optimal MLE on several datasets is a genuinely surprising and useful finding for the community, as the authors themselves acknowledge (Section 4.4). This is the kind of empirical observation that reshapes baselines.

- **Per-column loss tracking has concrete diagnostic value.** Figure 4 illustrates that individual column losses diverge substantially during training (Occupancy vs. Median Income on the House dataset), which can guide practitioners toward better preprocessing or targeted data collection. This is a qualitative advantage over black-box generative models, even if it follows architecturally from the design.

- **Plain MH's superiority over Tab-DDPM on continuous regression targets (House dataset) is well-supported.** Figure 2 illustrates that Tab-DDPM's quantization of continuous targets is a real limitation, and Plain MH's ability to generate continuous target values is a substantive advantage clearly demonstrated.

---

## Weaknesses

### Fatal
None.

### Major

- **"Gated" MoE terminology is misleading.** The paper's routing mechanism is a hard deterministic assignment: column *i* is always processed by expert *i*, with no learned gating network or mixture weighting. Standard Gated MoE (Shazeer et al., 2017) uses a trainable router to compute soft or sparse mixtures over experts per token. The paper cites Shazeer but never specifies what its gating function is, leaving readers to infer from Section 3.3 that routing is purely column-identity-based. This mischaracterization matters because it overstates similarity to the broader MoE literature and prevents accurate reproducibility. The method should be described as "hard-routed, column-specific expert layers" or the gating mechanism must be formally specified.

- **Parameter count confound in Claim 2 is unaddressed.** Table 3 compares NT DGPT-2 (80M parameters) to MH DGPT-2 (270M parameters)—a 3.4× increase. The gain in MLE (0.474 → 0.525) may reflect raw capacity rather than architectural superiority. No ablation against a non-Tabby model of equivalent parameter count is provided. As presented, Claim 2 conflates "MoE routing is effective" with "more parameters help," which are distinct hypotheses.

- **MMLP and MMLP-MH failures are catastrophic and unexplained.** Across the regression datasets in Table 2, MMLP and MMLP-MH variants frequently produce R² = 0.00 or asterisked sampling failures, while MH variants on the same tasks succeed. The paper notes this briefly ("Plain-trained MMLP models receive poor MLE scores on regression datasets") but provides no analysis, hypothesis, or ablation. This is not a peripheral result—MMLP is one of the two named Tabby variants and it fails reliably on half the dataset types. Without explanation, practitioners cannot know when to use or avoid MMLP.

- **Factual errors in the conclusion undermine confidence in manuscript quality.** The conclusion states: *"Tabby reaches parity with non-synthetic data in two out of three evaluated datasets, according to machine learning efficacy with a **Decision Tree Classifier**."* The paper uses a **Random Forest** throughout (Section 4.0.3 explicitly). Additionally, the paper evaluates six datasets, not three, making "two out of three" impossible to reconcile. These are not typos—they are substantive misstatements about the evaluation methodology and scope.

- **Inconsistent parity claims across the document.** The abstract states parity on "3 out of 6 datasets"; the introduction repeats "3 out of 6"; Section 4.1 and the Table 2 caption claim "4 out of 6." These are not trivially reconcilable (e.g., as "parity with real data" versus "highest MLE among all methods"), and the paper never clarifies the discrepancy. The numerical support for the 4/6 claim is also questionable: by inspection of Table 2, Tabby (any variant) is the best-performing method in approximately 3 datasets (Diabetes, Adult, House), with Tab-DDPM outperforming on Travel, Abalone, and Rainfall.

### Minor

- **Scalability with column count is unaddressed.** Replacing one block with V copies multiplies that block's parameters by V. The paper does not discuss what happens on wide tables (e.g., 50–500 columns), nor does it propose strategies such as clustering columns into shared experts. This is a real architectural limitation for many real-world tabular datasets.

- **Claim 3 is weak as a standalone evaluated claim.** Per-column loss tracking is architecturally inherent—it is not a separate contribution that requires empirical validation, but rather a favorable side-effect of the design. The experiment in Section 4.3 demonstrates that column losses differ, which is expected by construction. No actionable result is derived from this observation beyond qualitative description.

- **Discrimination trade-offs vs. Tab-DDPM not discussed.** Tab-DDPM achieves substantially better discrimination on Travel (1.4 vs. 3.0) and Adult (0.9 vs. 9.8) compared to Plain MH. While MLE is the primary metric, this gap warrants at least a brief discussion of the fidelity/diversity trade-off rather than leaving it implicit.

### Tiny

- The sequential column-by-column training process (Section 3.3) lacks a pseudocode box; this makes exact reproducibility harder than necessary given the paper claims to be the first to introduce this approach.
- Three runs provide limited statistical confidence for datasets with high variance (e.g., Rainfall Plain NT: 0.41 ± 0.35); a few key comparisons would benefit from significance testing.

---

## Nice-to-Haves

- **Ablation over which transformer block layer benefits most from MoE replacement.** Only the MLP block and LM head are tested; a sweep over block position would clarify where column-specific parameters are most useful.
- **Efficiency and compute comparison.** Training time and memory cost of MMLP and MH variants relative to NT, especially given the 3.4× parameter overhead for MH.
- **Privacy analysis.** Tabular synthesis is frequently motivated by privacy-preserving data sharing; while not required for a systems/architecture paper, a brief discussion of whether dedicated column parameters change memorization risk would strengthen real-world relevance.
- **Evaluation on larger/more diverse datasets.** The paper itself recommends (Section 4.4) moving to more challenging benchmarks; including even one larger or higher-dimensional dataset would preempt this concern.
- **Downstream model variety for MLE.** Currently only random forests are used; evaluating with gradient-boosted trees (XGBoost/LightGBM) as a second model would show robustness of the MLE claims, though this is not required given the metric's standard usage.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"Cherry-picked" abstract claim of "up to 7% improvement"**: The "up to" qualifier makes this phrasing accurate as a bound. Not a real misrepresentation.
- **Criticism that Claim 2 is evaluated on "only one dataset"**: The authors explicitly state this is an illustrative comparison of the architectural effect across model sizes, not a claim to be generalized. The scope is appropriate.
- **Criticism of small dataset sizes**: Dataset sizes (Diabetes N=576, Travel N=715, etc.) are standard and widely used in the tabular synthesis literature. This is not a paper-specific weakness.
- **"First architecture modification" claim lacks justification**: The paper provides adequate citation of prior works to establish they are training-technique-based; the claim stands as written.
- **Complaint that Tab-DDPM is not highlighted as best discrimination performer**: The paper's highlighted rows are defined by role (best prior work, best LLM, best Tabby), not purely by metric value. The discrimination advantage of Tab-DDPM is visible in the table and the design of the highlight system is defensible.

---

## Novel Insights

The most genuinely novel insight across the three reviews—not present in the paper's own contributions section—is the **tension between MoE routing granularity and task type**: the MH variant succeeds across classification and regression while MMLP fails catastrophically on regression. This pattern suggests that column-specific routing is beneficial only at the output interface (LM head), where column identity is directly relevant to prediction, while injecting column-specific routing deep in the shared representation (MLP blocks) may disrupt the inter-column feature extraction the earlier layers are responsible for. This is not analyzed in the paper and represents a mechanistic question that, if investigated, could substantially clarify when and why column-specific experts are helpful.

---

## Suggestions

1. **Rename and precisely define the routing mechanism.** Replace "Gated MoE" with a term that accurately reflects hard, deterministic column-index routing (e.g., "column-partitioned experts" or "hard-routed column experts"). Provide one equation or a pseudocode line defining exactly how column index maps to expert index during both training and inference.

2. **Add a parameter-matched baseline for Claim 2.** Include either a non-Tabby DGPT-2 with additional layers/width to reach ~270M parameters, or a LoRA-augmented DGPT-2 with equivalent parameter count. This is the single most important ablation missing from the paper.

3. **Investigate and explain MMLP failure.** Hypothesize and test whether the failure is due to gradient instability, loss of shared representations, or training instability. At minimum, show the loss curves for a failing MMLP run to characterize the failure mode.

4. **Correct the conclusion.** Fix "Decision Tree Classifier" → "Random Forest" and "two out of three" → the correct count consistent with the rest of the paper. Reconcile the 3/6 vs. 4/6 parity discrepancy throughout the manuscript with a clear definition of what each count refers to.

5. **Add a scalability discussion section.** Describe the linear parameter scaling with V, the practical column-count ceiling for the current architecture, and propose one concrete mitigation strategy (e.g., column clustering into K < V shared experts).

---

**Evaluation:**
- *Novelty*: Moderate-to-high — the per-column expert routing idea is genuinely new in this setting, but the "gated" framing overstates its relationship to the broader MoE literature.
- *Technical soundness*: Moderate — the MH variant is well-motivated and clearly described; the MMLP failures and the parameter confound in Claim 2 leave meaningful gaps.
- *Empirical support*: Moderate — Plain MH results are consistent and the Plain training insight is solid; however, inconsistent claims, unexplained failures, and 3-run statistics on high-variance small datasets limit confidence.
- *Significance*: Moderate — the paper opens a genuine new direction (architecture-level modification for tabular LLMs) and makes a practically useful finding (Plain training), but current breadth and depth of evaluation fall short of establishing the method's reliability.
- *Clarity*: Fair — generally readable, but the conclusion errors, inconsistent parity counts, and underspecified routing mechanism are substantive issues, not surface polish.

# Actual Human Scores
Individual reviewer scores: [5.0, 3.0, 1.0]
Average score: 3.0
Binary outcome: Reject
