Now let me search for calibration papers to anchor my score.Now I have enough calibration context. Let me write the final review.

**Summary of calibration papers:**
- CTSyn (tabular foundation model, cross-table diffusion): **Accept, scores 6,6,5,6** — novel architecture, cross-table generalization, strong baselines
- GADGET (tabular generation training method): **Reject, scores 6,1,3,5** — rejected due to questionable claims and one harsh reviewer
- MixturePFN (MoE for tabular ICL): **Accept, scores 6,8,6** — evaluated on 36 datasets with statistical significance against 19 baselines
- MixNAM (MoE for additive models): **Reject, scores 3,6,6,3** — limited evaluation

Tabby sits below CTSyn (which has cross-table generalization, stronger foundation model claims backed by pretraining) and MixturePFN (36 datasets, 19 baselines, statistical significance). Tabby is better than GADGET (which had a "strong reject" from one reviewer). The lack of parameter-controlled baseline and limited dataset/scaling evaluation is a significant weakness.

---

## Summary
Tabby introduces a post-training architectural modification for transformer-based LLMs that replaces designated MLP blocks and/or the language modeling head with Mixture-of-Experts layers where each expert is statically assigned to a specific data column. Applied to Distilled-GPT2, the best-performing Tabby variant (Plain MH) achieves MLE performance at or above the real-data ceiling on 3/6 benchmark datasets, and outperforms prior LLM-based tabular synthesis approaches on 4/6 datasets. The paper also introduces Plain training as a competitive baseline and demonstrates per-column loss tracking as a diagnostic tool.

## Strengths

- **Novel column-specific MoE head for tabular synthesis**: The idea of dedicating separate LM head parameters to each column is clean, well-motivated, and a genuine architectural contribution to the tabular synthesis space. It directly addresses the structural heterogeneity of tabular data in a way that generic LLM fine-tuning cannot.

- **Broad experimental comparison**: The paper evaluates against CTGAN, TVAE, Tab-DDPM, GReaT, TapTap, and Tabula across 6 datasets with both MLE and discrimination metrics, averaged over three seeds. This is a comprehensive comparison by the standards of the field.

- **Surprising and valuable Plain training finding**: The discovery that Plain-trained DGPT-2 is a highly competitive baseline—sometimes outperforming GReaT, GTT, and other specialized LLM training pipelines—is an important empirical contribution that could redirect future work. The paper acknowledges this honestly (Section 4.4).

- **Per-column loss tracking (Claim 3)**: Section 4.3 demonstrates a genuine practical benefit of Tabby's per-column training formulation—tracking individual column loss curves during training—yielding interpretable diagnostics not easily available from non-Tabby models (Figure 4).

- **Competitive performance on classification datasets**: Plain-trained Tabby MH achieves at or above the real-data ceiling on Diabetes (74.3±0.4 vs. original 73.3), Travel (87.7±1.2 vs. 87.5), and Adult (84.5±0.2 vs. 84.5), and achieves the highest MLE on House (0.75 vs. original 0.81).

---

## Weaknesses

### Fatal
None.

### Major

- **No parameter-controlled baseline makes the core architectural claim unverifiable.** Tabby MH increases DGPT-2 from 80M to 270M parameters (Table 3), a 3.4× increase. The paper attributes performance gains to the column-specific expert structure, but without comparing against a non-Tabby DGPT-2 of equivalent size (e.g., GPT-2 Medium/Large at ~345M parameters), it is impossible to determine whether the improvement comes from the MoE design or simply from having more parameters. This is the single most critical gap in the paper's evidence base, as it directly undermines Claim 1.

- **Expert specialization is never verified.** The central mechanism claim is that dedicated per-column experts allow each column's distribution to be better modeled. Yet the paper provides no evidence that the experts actually specialize by column—no activation analysis, gradient signal visualization, or any comparison of expert weights across columns. Without this, the architectural motivation is an assertion, not a demonstrated mechanism.

- **Claim 2 is supported by only one dataset subset, and results are negligible for large models.** The Llama vs. DGPT-2 scaling experiment (Section 4.2, Table 3) is conducted on a single 5160-row subset of one dataset. Tabby MH's improvement on Llama is 0.560 → 0.562 MLE (within error bars) with *worse* discrimination (24.2 → 25.3). This is insufficient evidence to generalize that "Tabby allows smaller models to achieve fidelity more similar to larger ones," and the Llama results actively contradict the claim for large base models.

- **MMLP and MMLP-MH variants are inconsistent and unexplained.** Two of the three proposed Tabby variants (MMLP, MMLP-MH) frequently perform worse than the NT baseline—e.g., Plain MMLP on Adult (77.4 vs. 84.5), MMLP-MH on Diabetes (68.1 vs. 73.3), and MMLP-MH producing R²=0.00 on House. The paper does not analyze why adding MoE to MLP layers is detrimental, nor provide guidance on when to use which variant. This inconsistency is unresolved and limits the contribution to a single working variant.

### Minor

- **"Gated Mixture-of-Experts" terminology is misleading.** Section 3.1 defines Tabby's MoE as column i always routing to expert L_{a,i} — fixed at construction, never learned. There is no gating network, soft routing, or competition among experts. Calling this "Gated MoE" conflates it with the standard MoE literature (Shazeer et al., 2017) in a confusing way. A more precise term (e.g., "column-partitioned heads" or "static per-column experts") would be more accurate and would not inflate the claimed connection to the MoE literature.

- **Many "improvements" are within error bars.** Several Tabby MH improvements over NT are statistically indistinguishable: Adult Plain MH 84.5±0.2 vs. NT 84.5±0.4; Abalone Plain MH 0.47±0.01 vs. NT 0.46±0.01. The paper does not apply significance testing to any paired comparison, making it difficult to trust claimed improvements on easier datasets.

- **Regression dataset performance gaps are substantial and understated.** On Abalone (0.47 vs. original 0.53), Rainfall (0.58 vs. 0.70), and House (0.75 vs. 0.81), Tabby MH does not achieve parity with real data, and Tab-DDPM often outperforms Tabby on these datasets. The abstract's claim of "achieving performance near or equal to that of real data" applies only to the three classification datasets and should be scoped accordingly.

- **Conclusion contains factual errors.** Section 5 states "Tabby reaches parity ... according to machine learning efficacy with a *Decision Tree Classifier*," while the paper uses a **Random Forest** throughout. The conclusion also states "two out of three evaluated datasets," while Section 4.1 and Table 2 caption report "4/6" or "3/6" parity results. These are concrete inconsistencies in the reported findings.

- **LTM framing is overstated relative to experimental scope.** The paper positions Tabby as "an initial step towards LTMs," but all experiments are on single, flat, small datasets with fixed schemas. Tabby's design is intrinsically tied to a fixed column count V, with no path demonstrated for heterogeneous or schema-varying multi-table settings. The paper does explicitly call this "an initial step," which is fair—but the abstract and introduction could more clearly scope the practical limitations.

- **Instability with GReaT/GTT is acknowledged but unexplained.** Multiple configurations fail to produce valid samples on Rainfall under GReaT and GTT training, including the base NT GReaT model and several Tabby variants. The paper attributes this to "pre-existing training techniques introduc[ing] undesirable effects" but does not analyze root causes (tokenization issues, sampling failures, distribution mismatch). This incomplete analysis leaves open whether Tabby robustly solves the problem or simply avoids it by using Plain training.

### Trivial

- **Conclusion uses "Decision Tree Classifier" where "Random Forest" is correct** — a clear editing error that should be fixed before publication.

- **Abstract cherry-picks the "up to 7% improvement" framing** — reporting median improvement across datasets would be more representative.

---

## Nice-to-Haves

- **Memorization/privacy analysis**: Given that LLM tabular methods serialize rows as text, trivial training example memorization is a real concern. Nearest-neighbor distance metrics or membership inference checks would strengthen the paper's practical claims.

- **Scalability analysis with column count**: Since Tabby adds V experts per MoE layer, parameters scale linearly with the number of columns. For tables with many columns, this could be prohibitive. An analysis of computational cost as a function of V would help practitioners.

- **Ablation on MoE layer placement**: The paper fixes the MoE block at the LM head (or a single MLP block) without studying whether placement depth or number of MoE layers matters. Even a simple two-point comparison (first block vs. last block vs. all blocks) would better justify the design.

- **Per-column comparison against non-Tabby loss tracking**: Figure 4's per-column loss tracking is a genuine Tabby benefit, but demonstrating how it differs from per-column loss computed by post-hoc masking on a standard NT model would better isolate what Tabby uniquely enables.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

1. **Harsh Critic: "Contradiction with LTM goal because schema is fixed"** — The paper explicitly frames Tabby as an "initial step" toward LTMs and acknowledges it is schema-tied. This is an honest scoping, not a contradiction. The LTM framing is overstated but not a misrepresentation sufficient to remove. *Kept as a minor weakness.*

2. **Harsh Critic: "MMLP forward-pass ambiguity undermines reproducibility"** — The paper describes the training process in sufficient detail in Sections 3.2 and 3.3; the sequential per-column training is clearly defined. This is a nitpick about forward-pass notation, not a genuine reproducibility concern. *Removed per hard rule on reproducibility nitpicks.*

3. **Harsh Critic: "Plain per-column loss is not unique to Tabby"** — While technically true that one could compute per-column losses from a standard LM by masking, the paper's point is that Tabby's architecture makes this a natural, built-in feature. The criticism does not undermine the claimed practical benefit. *Removed as a strawman.*

4. **Human Finder: Missing related works (TabSyn, etc.)** — Cannot independently verify the existence or exact relevance of such works. *Removed per hard rule on missing related work.*

5. **Harsh Critic: "Is 'gated' a factual error in citing MoE?"** — The paper does cite Shazeer et al. (2017) correctly; the terminological concern is about the label "gated MoE" for what is static routing. *Kept as a minor weakness about terminology, but not cited as a factual error about the MoE literature.*

6. **Neutral Reviewer: "Limited discussion of computational cost"** — Valid practical concern but not a core flaw; moved to Nice-to-Haves.

7. **Human Finder: Privacy/memorization analysis** — Valid but not standard for LLM tabular work at this stage; moved to Nice-to-Haves.

---

## Novel Insights

The paper's most genuinely novel empirical observation—surfaced across multiple reviewers—is that Plain training (sequence-order, no column shuffling, no label-guided prompting) with Distilled-GPT2 is surprisingly competitive against GReaT, GTT, and other specialized pipelines, sometimes outperforming them while being simpler and more stable. This finding, which the paper itself identifies and recommends as a baseline for future work, may be the paper's most impactful contribution regardless of the Tabby architecture. The per-column loss diagnostic also provides a useful and underexplored tool for understanding LLM tabular behavior.

---

## Suggestions

1. **Add parameter-matched dense baseline**: Train a 270M-parameter dense GPT-2 (e.g., GPT-2 Large at 345M, or a pruned variant) using identical training and evaluate on all 6 datasets. This single experiment would either confirm or refute whether the column-specific structure drives the gains.

2. **Verify expert specialization**: Show (e.g., via gradient norms, weight divergence, or activation patterns) that the column-specific experts actually develop distinct specializations over fine-tuning. Without this, the mechanism story is unverified.

3. **Expand scaling experiment to multiple datasets**: Run the DGPT-2 vs. Llama comparison on at least 3 datasets to support Claim 2 as a general finding rather than a single-dataset observation.

4. **Fix conclusion errors**: Replace "Decision Tree Classifier" with "Random Forest" and reconcile the "two out of three" wording with the actual 3/6 or 4/6 claims in the main text.

5. **Clarify terminology**: Rename "Gated MoE" to a more accurate description (e.g., "column-partitioned experts" or "static column-expert assignment") to avoid conflation with learned-gating MoE literature.

6. **Include significance testing**: For paired comparisons of key results (Tabby MH vs. NT), report p-values or confidence intervals on the *differences*, particularly for classification datasets where margins are small.

---

## Score and Decision

**Calibration:**

- **CTSyn** (tabular diffusion foundation model, cross-table generalization, pretraining on multiple tables): **Accepted, avg ~5.75.** Stronger scope, true cross-table generalization, solid baselines.
- **MixturePFN** (MoE for tabular ICL): **Accepted, avg ~6.7.** Evaluated on 36 datasets vs. 19 baselines with statistical significance, clearly beats baselines.
- **GADGET** (Gaussian-decomposition training for tabular synthesis): **Rejected, avg ~3.75.** Broader evaluation than Tabby but with one "strong reject" for conceptual issues.

Tabby is a legitimate contribution: it proposes a concrete architectural idea, runs a wide comparison, and produces an interesting empirical finding (Plain training). However, it falls meaningfully short of CTSyn and MixturePFN in key dimensions:
- Lacks the parameter-controlled baseline that is needed to attribute gains to architecture rather than scale.
- Claim 2 (scaling benefit) is unsupported by a single dataset with negligible Llama gain.
- Two of three proposed variants (MMLP, MMLP-MH) frequently fail without adequate explanation.
- Only 6 small datasets vs. 36 for MixturePFN; no statistical significance.
- No expert specialization verification.

On the positive side, Tabby is better than GADGET: it has real positive results on at least the classification datasets, a clean idea, and the Plain training finding is genuinely valuable. I place it between GADGET (rejected ~4) and CTSyn/MixturePFN (accepted ~5.75-6.7). Given the unresolved parameter-scaling confound as the central weakness, and the working variant being basically one out of three, I lean toward reject at this stage but not strongly.

**Score: 4.0**

The paper has a reasonable idea and real empirical wins, but the core architectural claim is not isolated from parameter scaling, two of three proposed variants underperform, and Claim 2 is supported by insufficient evidence. The paper is not fundamentally broken but needs substantive additional experiments—particularly the parameter-matched baseline—before the claims can be trusted.

MY FINAL SCORE: <pineapple>4.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>