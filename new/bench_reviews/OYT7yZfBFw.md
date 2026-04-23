Now I have enough information to write the meta-review. Let me synthesize everything.---

## Summary

TrajGPT introduces a Selective Recurrent Attention (SRA) mechanism with data-dependent decay for irregularly-sampled time-series pre-training, interprets the model as a system of discretized ODEs, and leverages this connection to enable time-specific inference that directly predicts arbitrary target timesteps. The model is evaluated on two large-scale healthcare EHR datasets (PopHR, 489K patients; eICU, 139K patients) across forecasting, drug prediction, and phenotype classification, demonstrating competitive zero-shot and few-shot performance against 17+ baselines.

---

## Strengths

- **Time-specific inference provides a clear, measurable benefit**: The ablation (Table 3) shows time-specific inference outperforms auto-regressive inference by 6.2% (71.7% vs. 65.5% top-10 recall), which is the single largest componentwise gain in the study. This is the paper's most substantive technical contribution — the ODE interpretation is not just cosmetic, it directly enables a practically useful inference procedure.

- **Strong zero-shot performance across all tasks**: TrajGPT achieves the best zero-shot result on all tasks evaluated — 67.2% (insulin), 72.8% (CHF) on PopHR (Table 1), and 45.1% (sepsis) on eICU (Table 2) — outperforming all baselines including models specifically designed for irregular time series. Zero-shot capability is meaningful in low-resource clinical settings.

- **Comprehensive evaluation breadth**: Experiments span two real-world EHR datasets with distinct characteristics (longitudinal population claims vs. ICU records), three learning regimes (zero-shot, few-shot, fine-tuning), and multiple task types with 17+ baselines. This is substantially broader than comparable work such as TimelyGPT.

- **Clinically coherent trajectory visualizations**: Figure 4 case studies show plausible disease progression dynamics (e.g., chronic IHD, hypothyroidism, and obesity preceding diabetes onset at ages 59–62), demonstrating that the learned representations capture medically meaningful comorbidity structure.

- **Efficient architecture**: O(N) training and O(1) inference complexity (Section 3.3) provides practical advantages for large EHR sequences over ContiFormer (ODE-based, quadratic) and standard Transformers.

---

## Weaknesses

### Fatal
None.

### Major

- **Asymmetric pre-training weakens baseline comparisons**: TrajGPT uses its natural next-token prediction pre-training. For other Transformer models without an established pre-training paradigm (Informer, Autoformer, FEDformer, PatchTST, TimesNet, ContiFormer), the paper applies a zero-masking procedure (Section 4.4: "randomly masking 40% of timesteps with zeros"). Zero-masking with zeros is a weaker self-supervised objective than next-token prediction: it provides no gradient signal from non-masked positions, does not optimize a natural sequence likelihood, and is not used as those models' intended training paradigm. This asymmetry most seriously affects encoder-style models (Informer family, PatchTST, ContiFormer), handicapping their representations before the downstream evaluation even begins. Comparison to TimelyGPT and PrimeNet — which use their own established objectives — is fairer, but the paper never distinguishes which baselines have equalized pre-training and which do not. As presented, it is impossible to determine whether TrajGPT's gains over the Informer family reflect architectural quality or pre-training objective quality.

- **Most headline margins fall within overlapping standard errors**: Inspecting Table 1, TrajGPT at K=5 (57.4±3.2) is *lower* than TimelyGPT (58.2±3.7) — a directional reversal of the headline claim. At K=10, TrajGPT (71.7±2.6) leads TimelyGPT (70.3±3.1) by 1.4 points with overlapping intervals. At K=15, TrajGPT (84.1±2.4) beats MTand (83.7±1.9) by 0.4 points and HeTVAE (83.2±3.2) by 0.9 points — all within overlapping confidence bands. For CHF full fine-tuning, **mTAND (85.4±2.5) outperforms TrajGPT (83.9±2.0)**; the paper dismisses this by appealing to scalability rather than performance. In Table 2, MTand full fine-tuning (52.5±2.1) outperforms TrajGPT (51.3±2.4). No formal significance tests are reported. Given bootstrap standard errors of ±2–4%, the paper's claims of "superior performance" and "excels" are not supported for the majority of comparisons. The most defensible advantage is in zero-shot settings, which is the paper's strongest story.

- **SRA's novelty relative to existing linear attention mechanisms is limited and underarticulated**: The SRA parallel form — O = (QK^T ⊙ D)V with D_nm = b_n/b_m — is the standard cumulative-product decay structure shared by multiple existing linear recurrent attention architectures. The recurrent form with data-dependent gating γ_n is structurally analogous to Mamba's selectivity mechanism, which the paper includes as a baseline without clearly articulating what SRA adds beyond applying a similar mechanism to healthcare time series with irregular timestamps and ODE interpretation. The ablation shows that the data-dependent decay (vs. fixed decay) yields only 1.4% improvement (71.7% vs. 70.3%), which is within noise. The genuine novelty lies in the ODE-motivated time-specific inference, not in the SRA mechanism itself. The paper should be clearer about this.

### Minor

- **Zero-shot classification methodology is unspecified**: The paper reports zero-shot AUPRC for insulin prediction and CHF classification (Table 1) and shows UMAP visualizations (Section 5.1), but never states the actual decision rule used to map sequence embeddings to binary labels. Is it nearest-centroid in embedding space? Cosine similarity to a disease token embedding? A score threshold on token probability? Without this, the zero-shot results cannot be reproduced.

- **Incomplete ablation table**: Table 3 contains a literal "?" for the "TrajGPT (without Pre-training)" row in the auto-regressive inference column. A question mark in a results table is unexplained and suggests either an unfinished experiment or suppressed result. Without this entry, the contribution of pre-training to auto-regressive performance cannot be assessed.

- **Cherry-picked case studies**: The trajectory analysis in Section 5.3 reports top-10 recall of 90.1% (diabetes) and 84.7% (CHF) for two specific patients — substantially above the population-level recall of 71.7%. No description of patient selection criteria is provided, raising the possibility of favorable selection. Population-level recall distributions or randomly sampled examples would be more informative.

- **Temperature hyperparameter τ=20 is unjustified**: The choice of τ=20 in γ_n = Sigmoid(·)^(1/20) compresses the decay toward 1 regardless of input, which partially undermines the "selective forgetting" motivation. No ablation or justification is provided for this value.

### Trivial

- **Causal language about comorbidity is unsupported**: Section 5.3 states "TrajGPT successfully forecasts diabetes onset by identifying related metabolic and circulatory symptoms," which implies causal inference. A next-token prediction model identifies statistical associations, not causal pathways; this language should be softened.

---

## Nice-to-Haves

- **Out-of-distribution evaluation**: Pre-training on PopHR and evaluating on eICU (or vice versa) would directly test the "generalizable representations" claim. Currently both datasets are independent in-domain experiments. The paper acknowledges this limitation and lists it as future work, but it would substantially strengthen the submission.

- **Formal significance tests**: Given standard errors of ±2–4%, paired bootstrap or permutation tests on the forecasting recall would clarify which comparisons are statistically meaningful.

- **Ablation comparing fair pre-training objectives**: Adding one experiment where TimelyGPT is pre-trained with next-token prediction (as it is designed for) and compared directly to TrajGPT would isolate whether the SRA architecture adds value beyond the pre-training paradigm.

- **Continuous/multivariate time series evaluation**: The paper focuses exclusively on discrete diagnosis codes. Extending to ICU vital signs (continuous measurements) would broaden the applicability claim and is already identified as future work.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **"SRA is identical to RetNet"** (Harsh Critic): The reviewer claims SRA's parallel form is the retention mechanism from "RetNet (Sun et al., 2023)" which the paper did not cite. Removed per the rule against citing missing related works that cannot be independently verified.

- **"ODE connection is just standard SSM derivation"** (Harsh Critic): The paper explicitly provides a proof in Appendix C and derivation in Appendix D. The appendices are stripped by the parser but exist in the original submission. Removed per the rule against criticizing absent appendix content.

- **"GPT-2 ablation (—) for time-specific inference is unjustified"** (Harsh Critic): Removed as a misunderstanding. GPT-2 lacks the recurrent state structure required for time-specific ODE-based extrapolation; the "—" is architecturally justified, not an omission.

- **"Cross-entropy loss inconsistency for zero-masking baselines"** (Harsh Critic): The paper states "All Transformer models performed 20 epochs of pre-training with cross-entropy loss" — while the baselines use zero-masking reconstruction, cross-entropy is applicable as a reconstruction objective. This is a minor presentation ambiguity, not a factual error, and the underlying pre-training asymmetry concern is already captured under Major weaknesses.

- **Reproducibility concerns about hyperparameters and training logs**: Removed per the rule on trivial implementation details.

---

## Novel Insights

The most genuinely novel contribution of this paper is the combination of ODE-motivated time-specific inference with GPT-style pre-training on EHR sequences. The 6.2% ablation gap between time-specific and auto-regressive inference (71.7% vs. 65.5%, Table 3) demonstrates that directly conditioning on the actual target timestep — rather than stepping autogressively at regular intervals — provides a meaningful advantage specific to irregularly-sampled medical data. This is a practical design insight that extends beyond this paper: models with recurrent hidden states parameterized by continuous time can, in principle, skip to any target horizon in O(1) inference steps, a property that is particularly valuable in clinical settings where follow-up visits occur at irregular and clinically meaningful intervals. The paper's connection of linear recurrent attention to neural ODEs under ZOH discretization provides a principled theoretical grounding for this inference strategy, even if the SRA mechanism itself is architecturally similar to existing linear attention approaches.

---

## Suggestions

1. **Equalize pre-training across baselines**: At minimum, provide one experiment comparing TrajGPT vs. TimelyGPT where both use their natural pre-training objective, and separately report results for the encoder models that received zero-masking. This would clarify whether TrajGPT's edge is architectural or pre-training-related.
2. **Report significance tests**: Add paired bootstrap p-values for the primary forecasting comparisons in Tables 1 and 2.
3. **Specify zero-shot classification rule**: State precisely how sequence embeddings are mapped to binary labels for the zero-shot tasks.
4. **Replace the "?" in Table 3**: Either complete or explain the missing ablation entry for "TrajGPT (without Pre-training) + auto-regressive."
5. **Provide population-level trajectory statistics**: Supplement the two case studies in Section 5.3 with a distribution of per-patient top-10 recall across the test set, or show randomly sampled patient trajectories.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Human Score | Comparison |
|---|---|---|---|
| TimelyGPT (close predecessor) | `/human_reviews/2sCcTMWPc2.md` | 5.5, Reject | TrajGPT extends TimelyGPT with time-specific inference and data-dependent decay; structurally similar but with broader eval and one genuine technical addition (time-specific inference) |
| MOTOR (EHR foundation model) | `/human_reviews/NialiwI2V6.md` | 7.5, Accept Spotlight | Much stronger novelty (first survival foundation model), larger scale, public release; TrajGPT is notably weaker |
| Context Clues (EHR evaluation) | `/human_reviews/zg3ec1TdAP.md` | 7.0, Accept Poster | Primarily evaluation work, praised for systematic benchmarking; TrajGPT has more methodological content but weaker statistical rigor |
| XTSFormer (irregular clinical events) | `/human_reviews/mH3yfzIPsL.md` | 5.0, Reject | Similar problem setting, comparable level of incremental contribution |
| qU1GtrDDst (weak novelty paper) | `/human_reviews/qU1GtrDDst.md` | 1.8, Reject | Much weaker; TrajGPT clearly above this anchor |

**Reasoning**: TrajGPT is clearly above the low anchor (1.8) and slightly above XTSFormer (5.0). Its most direct comparator is TimelyGPT (5.5, Reject), which it genuinely extends with time-specific inference, better zero-shot performance, and broader experimental coverage. However, TrajGPT shares TimelyGPT's core weaknesses: incremental architecture, asymmetric pre-training comparisons, and margins within statistical noise. MOTOR (7.5) represents a paper with clear novelty claims well-supported by experiments — TrajGPT does not reach this bar. The paper's strengths (time-specific inference, strong zero-shot, broad evaluation) place it above TimelyGPT, but the two major methodological issues (asymmetric pre-training, statistical significance) prevent a confident accept. I place it at **5.0** — above the borderline for TimelyGPT but below the bar set by stronger EHR foundation work.

**Axis evaluations:**
- *Originality*: Moderate — SRA is architecturally similar to existing linear attention; time-specific inference is the genuine contribution.
- *Importance of research question*: High — irregular EHR pre-training is a well-motivated and practically important problem.
- *Claims well-supported*: Weak — key claims of superiority are not statistically supported; the strongest claim (zero-shot) is reasonably supported.
- *Soundness of experiments*: Moderate — good breadth, but asymmetric pre-training and missing significance tests are real methodological gaps.
- *Clarity of writing*: Good — the paper is well-organized, though the zero-shot decision rule and ablation table issues are notable lapses.
- *Value to research community*: Moderate — the time-specific inference idea is practically useful; the broader framing as an EHR pre-training method is incremental.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>