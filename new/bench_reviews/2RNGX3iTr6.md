Now I have enough calibration. Let me synthesize the final review.

## Summary

The paper introduces Tabby, a post-training architecture modification for transformer-based LLMs that applies Mixture-of-Experts (MoE) layers—either to the transformer MLP blocks (MMLP), the language modeling head (MH), or both (MMLP-MH)—to dedicate expert parameters per tabular column. The core idea is that MoE specialization enables better modeling of individual column distributions and cross-column relationships. The paper also introduces "Plain" training as a simple baseline, compares across GAN, VAE, diffusion, and LLM-based methods on 6 datasets, and highlights per-column loss monitoring as a diagnostic benefit.

## Strengths

- **The MH variant consistently improves over the non-Tabby (NT) baseline across most datasets.** Table 2 shows Plain Tabby MH outperforms Plain NT on Travel (87.7 vs 85.5), Rainfall (0.58 vs 0.41), and House (0.75 vs 0.70), while matching on Adult (84.5) and Diabetes (74.3 vs 75.3). This demonstrates the MH design provides genuine gains over the non-Tabby LLM baseline.

- **Effective on the challenging Rainfall dataset where prior LLM methods fail.** Table 2 shows GReaT NT fails to produce valid samples in any run (N/A*), while Plain Tabby MH achieves the best MLE among LLM approaches (0.58) — demonstrating the architecture modification enables modeling of distributions that break prior LLM-based approaches.

- **Per-column loss tracking provides genuinely useful diagnostic information.** Figure 4 shows that the MoE training formulation natively yields per-column validation loss curves (e.g., revealing that Median Income barely improves while Occupancy converges), which is not available from standard LLM training or GAN/diffusion approaches. This supports Claim 3.

- **Valuable community observation about benchmark limitations.** Section 4.4 candidly notes that Plain training with Distilled-GPT2 achieves near-optimal MLE on several standard benchmarks (Adult, Diabetes), calling for more challenging evaluation datasets. This honest assessment is useful for the field.

- **Qualitative analysis of competing methods' failure modes.** Figure 2 provides a concrete, informative visualization showing Tab-DDPM only generates integer-valued regression targets on the House dataset — a clear artifact of that method's design.

- **The architecture modification is modular and composable.** Section 3.1 describes that Tabby replaces designated blocks after pretraining but before finetuning, making it compatible with any transformer-based LM and composable with existing training techniques (GReaT, TapTap, Tabula).

## Weaknesses

### Fatal

None.

### Major

- **Claim 1 is overclaimed relative to Table 2's full results.** The paper states "Tabby models achieve the highest MLE in 4 out of 6 datasets" (Section 4.1) and Claim 1 explicitly compares against "pre-existing tabular synthesis approaches" — which includes Tab-DDPM, CTGAN, and TVAE. However, looking at Table 2 comprehensively: Tab-DDPM achieves the highest MLE on Travel (88.9 vs. 87.7), Abalone (0.52 vs. 0.47), and Rainfall (0.60 vs. 0.58). Plain NT (not Tabby) achieves the highest on Diabetes (75.3 vs. 74.3). Tabby MH is the clear winner only on House and ties at ceiling on Adult. The "4 out of 6" figure only holds among LLM-based approaches, not against all methods as Claim 1 asserts. The paper should either revise Claim 1 to reflect the actual scope of the comparison or acknowledge the cases where diffusion-based methods remain competitive. This matters because the central claim's credibility is undermined by the gap between what is claimed and what the data shows.

- **The parameter-count comparison in Section 4.2 lacks a parameter-matched non-Tabby baseline, undermining Claim 2.** Table 3 compares 80M NT DGPT-2, 270M Tabby MH DGPT-2, 8B NT Llama, and 10.5B Tabby MH Llama. The Tabby MH DGPT-2 has 3.4× more parameters than the NT DGPT-2 — the MoE head expansion *is* the parameter increase. Claim 2 states Tabby "allows smaller LLMs to achieve synthetic data fidelity more similar to that of LLMs with higher parameter counts," but without a wider/deeper ~270M non-Tabby DGPT-2, one cannot distinguish the effect of the MoE architecture from the effect of simply adding more parameters. That a 270M model performs between an 80M and an 8B model is expected regardless of architecture. This matters because it directly undermines the paper's argument that the architectural modification itself is responsible for the gains.

- **The MMLP variant — one of the two core architectural contributions — frequently fails catastrophically, with no diagnosis.** Table 2 shows MMLP produces disastrous results on regression datasets: R² = 0.00 on House, 0.11 on Rainfall, 0.28 on Abalone (vs. NT at 0.70, 0.41, 0.46). The MMLP-MH combination also produces 0.00 on House. On classification, MMLP drops Adult accuracy from 84.5 (NT) to 77.4. The paper presents MMLP as a co-equal contribution alongside MH in Figure 1 and Section 3.1, but does not explain *why* adding column-dedicated experts to the MLP layer destroys performance. This matters because half the proposed architecture is actively harmful on multiple datasets, and the absence of any hypothesis or investigation into this failure undermines confidence in the paper's understanding of when and why the MoE approach works.

### Minor

- **The "up to 7%" improvement in the abstract is cherry-picked.** The abstract claims "up to 7% improvement compared to previous tabular dataset synthesis methods." This 7% appears to come from Tabby MH over NT on House (0.75 vs 0.70 ≈ 7%), which is a comparison against the authors' own baseline, not "previous methods." Against Tab-DDPM on other datasets, Tabby is worse. The framing is misleading.

- **Incompatibility with GReaT training is inadequately explained.** Table 2 shows GReaT + Tabby MH drops Diabetes MLE from 74.3 (Plain) to 63.7 and crashes Rainfall (0.00 on some runs). This incompatibility with the primary existing LLM training method is noted but not investigated, which limits the practical applicability of Tabby beyond Plain training.

- **The conclusion contains factually incorrect claims.** Section 5 states "Tabby reaches parity with non-synthetic data in two out of three evaluated datasets, according to machine learning efficacy with a Decision Tree Classifier." In reality: (a) 6 datasets were evaluated, not 3; (b) Random Forest was used, not Decision Tree; (c) parity was achieved on 3 classification datasets (Diabetes, Travel, Adult), not 2.

- **Section 4.2 uses only 5 epochs on a dataset subset, and compares LoRA (Llama) vs. full fine-tuning (DGPT-2).** These design choices limit the strength of the scaling analysis. The discrimination scores for Llama (24-25%) are much worse than DGPT-2 (16%), which is acknowledged but not adequately explained.

### Trivial

- Presenting "Plain training" as a contribution (Section 3.2 / 4.0.1) is unusual — it is standard autoregressive fine-tuning without any special tabular technique. However, the paper's observation that this simple baseline was absent from prior work is a valid point.

## Nice-to-Haves

- A parameter-matched non-Tabby baseline (~270M DGPT-2) would directly test whether the MoE architecture design rather than parameter count drives the improvement in Claim 2.
- Per-column generation quality analysis (beyond just loss) comparing Tabby MH vs. NT would directly test whether expert specialization improves column-specific modeling.
- Investigation into why MMLP fails catastrophically on regression datasets — even a hypothesis and preliminary experiment — would substantially strengthen or honestly limit the paper's scope.
- Evaluation on more challenging datasets where Plain NT clearly fails would make the benchmarks more discriminative.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The Plain training baseline often matches or beats Tabby"** — While Plain NT does beat Tabby MH on Diabetes (75.3 vs 74.3) and matches on Adult, this is framed by the harsh critic as undermining the paper's contribution. However, Tabby MH still consistently improves over NT across most datasets (Travel, Rainfall, House). The observation about benchmark difficulty is already acknowledged by the paper (Section 4.4) and is a valid finding, not a weakness of the method itself. WEAKENED to minor.

- **"Presenting Plain training as a contribution is unusual"** — The paper positions it as a useful baseline that prior work overlooked, not as a major technical contribution. This is a presentation nitpick. MOVED to Trivial.

- **"Paper should show generated sample rows"** — A nice-to-have visualization request but Table 2 and Figure 2 already provide quantitative and qualitative evidence respectively. MOVED to Nice-to-Have.

- **"Test on full GPT-2 (not Distilled-GPT-2)"** — Generic "evaluate on more models" request. The paper already tests two very different model families/sizes. MOVED to Nice-to-Have.

- **"Per-column MoE might prevent cross-column learning"** — This is speculative and not supported by evidence. The MH variant works well, suggesting cross-column learning via attention is preserved. MOVED to Nice-to-Have.

- **The Strength Finder's claim that Tabby MH "achieves the highest MLE among all methods on 4/6 datasets"** conflicts with verified Major weakness about Tab-DDPM outperforming on 3/6 datasets. REMOVED from strengths.

## Novel Insights

The paper reveals an interesting asymmetry: applying MoE to the LM head (MH) consistently helps, while applying it to the MLP layers (MMLP) catastrophically fails on regression tasks. This suggests that dedicating expert parameters at the output stage (where column-specific decoding happens) is beneficial, but dedicating parameters at the intermediate representation stage (MLP) may isolate column representations too early, preventing the cross-column attention from building useful joint features. This is a finding that could inform future work on MoE-based architectures for structured data, though the paper itself does not investigate this hypothesis.

## Suggestions

- Revise Claim 1 to clearly state the scope: "Tabby models achieve the highest MLE among LLM-based approaches on 4 out of 6 datasets, and outperform Tab-DDPM on 2 out of 3 regression datasets." Acknowledge Tab-DDPM's superiority on Travel, Abalone, and Rainfall MLE explicitly.
- Add a parameter-matched non-Tabby baseline (wider DGPT-2 at ~270M parameters) to Table 3 to isolate the architectural effect from the parameter-count effect for Claim 2.
- Investigate and discuss the MMLP failure mode — at minimum, offer a hypothesis for why column-dedicated MLP experts hurt regression performance while column-dedicated LM heads help. This could be the paper's most insightful contribution if properly analyzed.
- Correct the factual errors in the conclusion (number of datasets, classifier used, and number of parity-achieving datasets).

## Evaluation Axes

**Originality:** The idea of applying per-column MoE experts to the LM head for tabular synthesis is clean and reasonably novel. However, MMLP (the other contribution) fails, and the parameter-efficiency claim is undermined by missing controls. Moderate originality.

**Importance of research question:** Tabular data synthesis is practically important, and improving LLM-based approaches is a relevant direction. The observation about benchmark inadequacy is valuable.

**Claim support:** Claim 1 is overclaimed relative to full results. Claim 2 lacks necessary controls. Claim 3 is well-supported but minor. Claims are partially but not fully supported.

**Experimental soundness:** Comprehensive comparison across method families, but key methodological gaps (no parameter-matched baseline, MMLP failure unexplained, limited scaling analysis).

**Clarity:** Generally well-written with clear figures, but the conclusion contains errors and Claim 1's scope is ambiguous.

**Community value:** The per-column MoE head idea, the Plain training baseline observation, and the benchmark critique are all useful contributions, but the overclaimed results reduce trust.

## Calibration Anchors

- **TabSyn** (4Ay23yeuz0, avg 6.75, Accept oral): Tabular synthesis with strong, clearly best results across all baselines and metrics. This paper is weaker because its main claim doesn't hold against all baselines, and TabSyn has cleaner experimental methodology.

- **TabDAR** (kkGIbmpCHU, avg 4.75, Reject): Diffusion-nested autoregressive tabular synthesis with overclaimed results ("18-45% improvement") that doesn't consistently beat baselines. Similar pattern to this paper — strong claims undermined by own tables. Slightly higher than TabDAR due to Tabby having more honest discussion of limitations.

- **TAEGAN** (pBqOH2g6K1, avg 4.5, Reject): Tabular GAN for small datasets that doesn't beat all methods and has missing baselines. Similar level of overclaiming.

- **TabFMs** (hz2zhaZPXm, avg 3.5, Reject): LLM-based tabular foundation model with underperforming results and lacking baselines. This paper is better than TabFMs because it has clearer gains on some datasets and more honest discussion.

- **TDTransformer** (r8tMECbxOl, avg 5.0, Reject): Tabular domain transformer with overclaimed statements and questionable baseline fairness. Similar positioning.

- **MG-NeRF** (WKfMFtlz5D, avg 2.5, Reject): Proposed variant performs worse than baselines yet claims "appreciable achievement" — catastrophic failure of variant. This is an extreme; Tabby's MH variant works, only MMLP fails.

This paper sits in the 4.0–5.0 range based on anchors: clearly above the low-tier papers (~2.5-3.5) that have fundamental failures across the board, but below the medium-tier accepts (~6-7) where methods clearly beat all baselines. The MMLP failure and overclaimed results position it alongside TabDAR/TAEGAN rather than TabSyn.

## Score and Decision

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>