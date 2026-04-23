Now I have all the information needed. Let me compile the final review.

## Summary

This paper formalizes the online Federated Continual Learning (online-FCL) problem, where data arrives in mini-batches that can only be processed once, and proposes a memory-based approach using Bregman Information (BI) as an epistemic uncertainty estimator for sample selection. BI, derived from a bias-variance decomposition of the cross-entropy loss, is used to select representative (low-BI, bottom-k) or hard-to-forget (high-BI, top-k) samples for replay, addressing catastrophic forgetting in this online setting. The method is evaluated across vision and text tasks, demonstrating consistent forgetting reduction compared to FL baselines and generative FCL methods.

## Strengths

- **Online-FCL problem formulation (Section 3.1):** Identifying and formalizing that existing FCL methods assume offline access to complete task datasets is a legitimate and practical contribution. The problem is well-motivated with realistic scenarios (e.g., streaming data from edge devices, COVID-19 variant classification), and the formalization follows established notation from Qi et al. (2023).

- **Theoretically motivated memory management via Bregman Information (Section 3.2.1, Eq. 1):** Using BI derived from a bias-variance decomposition of cross-entropy as an epistemic uncertainty estimator is a principled and novel choice for memory-based sample selection in FCL. Figure 2 effectively illustrates why BI captures a different aspect of uncertainty than confidence scores — points near decision boundaries have low confidence but low BI (high data density), while outliers can have high confidence but high BI.

- **Consistent forgetting reduction across datasets (Tables 1–4):** BI-Bottom consistently achieves the best or near-best forgetting scores. On CIFAR10 M=1000, BI-Bottom achieves F=19.07 vs. next-best F=24.30 (ER); on CRC-Tissue M=120, F=7.99 vs. next-best F=8.71 (RC-Bottom); on text tasks, BI achieves the lowest forgetting across all datasets in Table 4 (e.g., F=26.97 on DBpedia vs. 27.97 for MS). The consistency of this pattern across domains and memory sizes supports the core claim.

- **Evaluation on medical imaging datasets with realistic class imbalance (Section 4.1, Table 3):** Going beyond standard CIFAR-only evaluations, the paper tests on CRC-Tissue and KC-Cell with inherent class imbalance, where classes are assigned to tasks based on class size. This provides more realistic assessment conditions than typical FCL benchmarks.

- **Modality-agnostic design through TTA-based uncertainty (Section 4.1):** The BI estimation framework adapts perturbation strategies across modalities (standard augmentations for images, Gaussian noise on latent embeddings for text), enabling evaluation beyond vision-only tasks — a genuine advantage over generative FCL methods that are restricted to images.

## Weaknesses

### Fatal
None.

### Major

- **Table 5 "BI (best)" is an asymmetric and misleading headline comparison.** Table 5 reports "BI (best)" — selecting the best configuration (memory size and Top/Bottom-k) for BI — while omitting the closest competitors (ER, CBR, other uncertainty scores) entirely. This creates a misleading impression of dominance. In the detailed tables, simpler methods frequently match or beat BI on accuracy: CRC-Tissue M=80: RC-Bottom (57.50) outperforms BI-Bottom (57.05); CIFAR10 M=500: RC-Bottom (31.17) beats BI-Bottom (27.84); 20NewsGroup: CBR (45.21) beats BI (44.92); Yahoo Answers M=100: CBR (79.78) beats BI (78.68). On KC-Cell M=160, ER (22.29) even outperforms BI's best (21.61) on accuracy. The paper should either report all methods' best configurations in Table 5 or clearly note that "BI (best)" selects across configurations while other methods are not given the same advantage. The current presentation inflates the headline result.

- **Evidence for BI superiority over simpler uncertainty scores on accuracy is weak and inconsistent.** While BI is consistently the best or near-best on forgetting, on accuracy it is frequently matched or beaten by simpler scores (RC, CBR) across multiple datasets and memory sizes. The margins between BI and alternatives are frequently within one standard deviation (e.g., CIFAR100 M=2000: BI-Bottom F=6.96 vs. MS-Bottom F=6.84; CRC-Tissue M=80: RC-Bottom F=12.91 vs. BI-Bottom F=13.63). The paper's narrative that BI "outperforms the considered memory-based baselines in most of the cases" (Section 4.2) overclaims, since this is primarily true for forgetting but not for accuracy. The paper should be more precise about the accuracy-forgetting tradeoff and where BI's advantages manifest.

### Minor

- **The generative baseline comparison is a trivially expected finding presented as a contribution.** Contribution #2 states "We highlight the limitations of current state-of-the-art generative-based solutions to work in the online setting." The paper acknowledges in Section 5 that the performance gap "stems from the different assumptions of the online scenario," and provides Figure 5 showing poor synthetic images. That methods requiring 100 epochs over data perform poorly with 1 epoch is predictable without experimentation. While this validates the motivation, presenting it as a standalone contribution inflates the paper's scope. — This inflates the claimed contributions but doesn't undermine the core technical contribution.

- **Text experiments use frozen pre-trained embeddings, limiting the cross-modality claim.** For text tasks, the paper uses e5-small-v2 (384-dim) as a frozen encoder with a single-layer MLP (Section 4.1). This tests whether BI-based memory management works on low-dimensional feature vectors, not whether it handles the challenges of online representation learning for text (tokenization, sequence modeling, catastrophic forgetting in the encoder). The "beyond vision" claim is partially undermined — the modality-agnostic property is demonstrated for a shallow classification regime, not for end-to-end learning where catastrophic forgetting is most severe.

- **Data inconsistency between Table 3 and Table 5 for KC-Cell.** Table 3 reports BI-Bottom M=160 F=59.85, but Table 5 reports KC-Cell F=59.05. This discrepancy, while small, raises concerns about data reporting accuracy in the headline table.

- **The "no information loss" claim (Section 3.2.1) is overclaimed.** The paper states that compared to other uncertainty scores, "there is no information loss in the estimation step." While BI's use of raw logits via LSE avoids the softmax normalization that discards magnitude information, the LSE aggregation in Eq. 1 is itself a specific functional form that compresses the full perturbed logit distribution. The advantage is real (operating on logit scale preserves more information than probability-scale metrics), but "no information loss" is too strong.

### Trivial
- None verified.

## Nice-to-Haves

- Inclusion of more sophisticated online-CL baselines (e.g., MIR, ASER, GSS, CBS) adapted to the federated setting would strengthen the evaluation, though the paper's argument that ER is surprisingly competitive (citing Soutif-Cormerais et al., 2023) provides reasonable justification for its absence.
- Statistical significance testing (e.g., Wilcoxon signed-rank over seeds) would help distinguish signal from noise given that many margins are within one standard deviation.
- A qualitative analysis of which samples BI selects vs. other scores (e.g., t-SNE/UMAP of memory buffers) would reveal whether the claimed epistemic-vs-aleatoric distinction manifests in practice.
- End-to-end text experiments where the encoder is also trained online would more convincingly test the cross-modality claim.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Class-based parameter averaging causes non-trivial privacy leakage"**: The paper explicitly addresses this in Section 3.2.2, noting that "if sharing the class information is not possible, we can rely on standard averaging strategies (e.g., FedAvg or FedProx) without hampering the performance." The flexibility is acknowledged and the fallback is provided.

- **"The burn-in period and jump parameter are engineering heuristics, not conceptual contributions"**: These are practical adaptations for the online setting, not claimed as deep conceptual contributions. The paper positions them appropriately.

- **"The assumption of non-overlapping classes for intra-client tasks limits generality"**: This is inherited from prior FCL work (Qi et al., 2023) and is standard practice in the field. Criticizing inherited assumptions is scope creep.

- **"Absolute performance numbers are strikingly low"**: The low absolute performance reflects the difficulty of the online-FCL setting and affects all methods equally. This speaks to problem difficulty, not a specific flaw of the proposed method.

- **"TTA-based BI estimates conflate sensitivity to augmentation choice with genuine epistemic uncertainty"**: The paper acknowledges this in the Limitations section and provides ablation studies on augmentation sets in Appendix A.7.

- **"Missing MIR, ASER, GSS, CBS baselines"**: The paper provides justification for using ER (citing empirical surveys showing it is surprisingly competitive). While these would strengthen the paper, their absence is a nice-to-have rather than a core flaw.

- **Formatting/typo nitpicks**: Including the KC-Cell F=59.05 vs 59.85 discrepancy is noted as a minor data issue, not a formatting issue.

## Novel Insights

The most insightful observation across the reviews is the tension between BI's consistent forgetting advantage and its inconsistent accuracy advantage. This suggests that BI-based bottom-k selection produces memory buffers that are particularly effective at preserving past-task performance (low forgetting) but not necessarily at improving overall classification accuracy. This is consistent with BI's theoretical property of capturing epistemic uncertainty: low-BI samples are "representative" in the sense of having low uncertainty about the data generating process, which helps maintain past-task boundaries, but may not select the most discriminative samples for current-task learning. The paper would benefit from explicitly analyzing this accuracy-forgetting tradeoff rather than conflating the two metrics in its narrative.

## Suggestions

- Replace "BI (best)" in Table 5 with a single predetermined configuration (e.g., BI-Bottom with the same memory size used for ER/CBR comparisons) OR add ER (best) and CBR (best) rows to provide a fair comparison.
- Add a discussion section or table explicitly comparing BI's forgetting advantage vs. its accuracy disadvantage relative to simpler methods, to make the tradeoff transparent.
- Downgrade contribution #2 from a standalone contribution to part of the motivation, and avoid listing the failure of offline generative methods in an online setting as a novel empirical finding.

## Score and Decision

**Calibration anchors:**

- **High (>7):** dOAkHmsjRX (7.50, Spotlight) — Budgeted online CL with strong empirical results and comprehensive evaluation; this paper is weaker due to mixed accuracy evidence and Table 5 asymmetry. TpD2aG1h0D (8.67, Oral) — Meta-CL with excellent theory and empirics; far above this paper's contribution level. 8EyRkd3Qj2 (7.50, Spotlight) — New FL formulation with multimodal validation; stronger validation across modalities than this paper.
- **Medium (4–6):** Xi7UoErFRt (5.0, Reject) — FedGP for CFL with cherry-picked comparisons; this paper shares the selective-presentation concern but has a clearer novel contribution (BI). nFI3wFM9yN (6.0, Poster) — New federated online formulation with theory; comparable novelty in formulation but less empirical depth. he4CPgU44D (4.75, Reject) — Active CL with interesting problem but weak baselines; this paper is slightly stronger with a concrete method and more thorough evaluation. ZMjflI1aL0 (4.33, Reject) — MSCL for online CL with imbalanced data; this paper is comparable in scope but has better baselines and more diverse evaluation.
- **Low (<3):** MeSfNZjGvN (2.5, Withdrawn) — FedPeWS with only FedAvg/FedProx baselines and overclaimed benefits; this paper is clearly above this level with multiple baselines and genuine contributions. ZyMXxpBfct (1.5, Reject) — Overclaimed catastrophic forgetting results with neglected prior work; this paper is substantially more honest and well-grounded.

This paper falls between the medium anchors. It has a genuine contribution (online-FCL formalization + BI-based memory management) with consistent forgetting advantages, but the evidence for BI's superiority on accuracy is weak, Table 5 presents a misleading headline comparison, and the generative baseline finding is trivially expected. Compared to FedGP (5.0, rejected for cherry-picked comparisons and questionable novelty), this paper has a clearer novel contribution but similar presentation concerns. Compared to the active CL paper (4.75, rejected for weak baselines despite novel problem), this paper is slightly stronger. The paper sits at the borderline — contributions are real but the evidence and presentation have notable issues.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>