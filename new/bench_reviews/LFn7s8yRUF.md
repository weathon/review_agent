Now I have a clear picture of the paper. Let me do calibration searches before writing the final review.Now I have sufficient calibration data. Let me write the final review.

---

## Summary

This paper investigates six data augmentation (DA) techniques — synonym replacement (SR), random insertion (RI), random swap (RS), random deletion (RD), back translation (BT), and LLM-based paraphrasing — for fine-tuning LLaMA3-8B with LoRA to create character-personalized Chinese dialogue models. The study uses two datasets (Paimon from *Genshin Impact* and Zhen Huan from *Empresses in the Palace*), evaluating performance via training/validation loss curves and BLEU/ROUGE scores. The main finding is that meaning-preserving methods (BT and SR) outperform meaning-distorting ones (RD, RI) and LLM paraphrasing.

---

## Strengths

- **Dataset-characteristic-aware explanation of DA effectiveness (Section 4.1.2):** The paper goes beyond surface reporting by connecting RD's poor performance on the Zhenhuan dataset to the fragility of classical Chinese idiomatic expressions (where removing a single word alters meaning), and explaining the failure of LLM paraphrasing on Paimon through SparkDeskV4's inability to handle game-specific world-building terminology. This contextual reasoning is the paper's most substantive contribution.

- **Dual evaluation perspectives (Figure 3 and Figure 4):** Analyzing both training/validation loss dynamics and text generation quality (BLEU/ROUGE) provides complementary evidence. The observation that Paraphrasing achieves relatively higher BLEU/ROUGE than RD despite overfitting in loss curves — explained by semantic preservation without diversity — is a non-trivial nuance that adds depth to the analysis.

---

## Weaknesses

### Fatal

- **No no-augmentation baseline.** The paper's stated goal is to show that DA improves personalized model training, yet every comparison in Figure 4 is *among* DA methods. A model trained on original, unaugmented data is never evaluated. Without this control, it is impossible to determine whether any DA method helps, hurts, or has no effect relative to the unaugmented setting. The paper's central premise — that DA is beneficial — is experimentally untested. This is not an optional ablation; it is the foundational comparison the study requires.

- **Abstract claims three datasets; paper uses two.** The abstract (line 17) explicitly states "we apply these techniques across three distinct datasets." The paper evaluates exactly two (Paimon and Zhenhuan). Section 5.1 itself acknowledges "only two datasets available." This is not a minor discrepancy — the generalization claims built on "three distinct datasets" in the abstract collapse in the body of the paper, and it raises questions about the paper's accuracy more broadly.

### Major

- **BLEU/ROUGE scores are implausibly high with no evaluation protocol described.** Figure 4 reports BLEU scores of 0.40–0.60 and ROUGE-1 of 0.50–0.70 for open-ended character dialogue generation. These values are extraordinary — state-of-the-art systems on open-ended dialogue benchmarks rarely exceed BLEU 0.10–0.15 in standard evaluation. The paper never describes: how the test set was constructed (size, overlap with training data), whether scoring is character-level or word-level (critical for Chinese), or what the reference responses are. Numbers this high without any methodological justification could indicate test-train overlap, incorrect metric application, or character-level tokenization that inflates scores. Without explanation, the quantitative results cannot be interpreted or trusted.

- **The core finding is already well-established in the literature the paper cites.** The conclusion — that meaning-preserving augmentations (BT, SR) outperform meaning-distorting ones (RD, RI) — is precisely what Wei & Zou (2019) established in the EDA paper, which the authors themselves cite. The paper presents no new method, no new theoretical framework, and no finding that extends or contradicts prior work. Reproducing a known result on two small, proprietary, single-language datasets does not constitute a novel research contribution at the level expected for ICLR.

- **Table 1 contains apparently incorrect benchmark numbers (Section 3.3).** Table 1 reports LLaMA3-70B at MATH=89.1 and LLaMA3-8B at MATH=85.6, and LLaMA2-8B at MATH=81.2. Published LLaMA3 evaluations place the 70B model at approximately 50 on MATH (pass@1). These numbers appear either fabricated or sourced from an unreliable origin. While this table is used only as background motivation for the hardware choice, incorrect baseline numbers undermine confidence in the paper's factual accuracy throughout.

- **Dataset sizes and splits are never reported.** The total number of dialogue turns, train/validation/test splits, and the ratio of augmented to original data are absent from the paper. This makes it impossible to interpret the loss curves (what is "convergence" without knowing dataset size?), the BLEU/ROUGE scores, or to assess whether the study has sufficient statistical power for any conclusion.

### Minor

- **LoRA hyperparameters not reported.** Rank, alpha, dropout, which weight matrices are adapted — none are specified. Given that the paper's analysis depends on LoRA-based fine-tuning, the training setup is insufficiently described for even qualitative interpretation.

- **Overfitting claims are based on visual inspection only.** The claim that RI "leads to overfitting" while RS "does not" rests entirely on visual comparison of loss curves (Figure 3) with no quantitative criterion (e.g., gap between training and validation loss at convergence, or early stopping epoch). Validation loss is also a weak proxy for dialogue quality.

- **Logical incoherence in Section 4.3.** The paper recommends against LLM-based paraphrasing partly on the grounds that "training a paraphrasing model requires a substantial amount of data." However, SparkDeskV4 is an off-the-shelf LLM that requires no training by the authors. The correct criticism — that generic off-the-shelf LLMs fail on domain-specific content — is present in the text but partially obscured by this confused framing.

### Trivial

- None beyond the issues noted above.

---

## Nice-to-Haves

- **Augmentation ratio ablation.** Whether and how much DA helps may depend critically on the ratio of augmented to original data, but this is never varied. Even a simple 2–3 point ablation would clarify the conditions under which any DA method is beneficial.
- **Human evaluation for character fidelity.** BLEU/ROUGE do not measure whether a model sounds like Paimon or Zhen Huan. A small human preference study would directly address the stated goal of replicating a character's tone and linguistic habits.
- **Qualitative output examples per DA method.** Side-by-side model responses under each DA condition would allow readers to judge whether metric differences reflect genuine character fidelity differences.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh Critic: Table 1 numbers are "fabricated."** While the numbers are suspicious, labeling them fabricated is too strong without certainty about the evaluation protocol used. Kept as a major concern about factual accuracy, not fabrication.

- **Strength Finder: "Systematic comparison of six DA methods across two linguistically distinct datasets as a core strength."** The comparison is incomplete (no no-augmentation baseline) and the datasets are only two despite the abstract's claim of three. This cannot stand as a genuine strength when it conflicts with a verified major weakness.

- **Strength Finder: "Focus on Chinese character-personalization for localized LLMs" as a standalone strength.** Generic motivation; does not distinguish this paper from any other Chinese-language LoRA fine-tuning paper. Removed.

- **Strength Finder: "Honest discussion of limitations."** Generic. The limitations section acknowledges only two datasets while the abstract still claims three — the acknowledgement is incomplete. Removed.

- **Strength Finder: "Multiple evaluation perspectives combining loss dynamics with generation quality."** This would be a strength if the evaluation protocol for BLEU/ROUGE were described. Given the unexplained and implausibly high scores, this structural choice cannot be credited as a genuine strength without verification.

---

## Novel Insights

None beyond the paper's own contributions. The dataset-characteristic explanation (classical Chinese idiom fragility under RD; game-specific terminology failure under LLM paraphrasing) is the most interesting observation in the paper, but it is qualitative and unverified by any augmentation-quality analysis.

---

## Suggestions

1. **Add a no-augmentation baseline as the first priority.** Without it, the paper cannot answer its stated research question.
2. **Describe the evaluation protocol completely**: test set size, construction method, whether it was drawn from the augmented pool or held-out from original data, and whether BLEU/ROUGE are character-level or word-level for Chinese text.
3. **Correct or remove Table 1.** The MATH scores do not match published LLaMA3 benchmarks and the table is used only to justify the hardware choice, which could be stated in one sentence.
4. **Correct the abstract** to reflect that two datasets were used, not three.
5. **Report dataset statistics**: total turns per dataset, train/val/test splits, and augmented:original data ratio.
6. **Report LoRA hyperparameters**: rank, alpha, dropout, adapted matrices.

---

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Regulating text augmentation level (NLP DA, proposed new method) | TkP2RtR4hr | **3.0** | Most topically similar. Proposed a novel method but was rejected for unclear scope, missing reproducibility, weak baselines. This paper is weaker: no novel method, missing control baseline, 3-vs-2 dataset mismatch. |
| FreeLM (NLP fine-tuning, overclaimed, weak) | qgLyKwXVDs | **2.0** | Rejected for unsupported overclaims. Similar pattern of weak evidence for strong claims. |
| LongLoRA (LoRA fine-tuning, strong accepted) | 6PmJoRfdaK | **7.0** | High-quality LoRA work with clear novel contribution and strong experiments. Incomparable — this paper has none of those properties. |
| Fine-tuning with Reserved Majority (LoRA variant) | ZV7CLf0RHK | **7.5** | Spotlight paper with rigorous ablations and novel method. Far above this paper's quality. |
| PingPong role-playing benchmark | 996aKQIom0 | **3.83** | Role-playing evaluation with methodological gaps. This paper's core structural flaws (missing baseline, suspicious metrics) are more severe. |

The paper under review shares the profile of TkP2RtR4hr (avg 3.0) but is arguably weaker: it proposes no novel method (TkP2RtR4hr at least introduced a technique), has a verifiable discrepancy between the abstract and the body, is missing the fundamental no-augmentation control, and reports BLEU/ROUGE values that cannot be interpreted without an evaluation protocol. The finding it reports is already established. I place this paper **below** the TkP2RtR4hr anchor.

**Overall assessment:** The paper reads as a preliminary technical report, not a research contribution suitable for ICLR. The two fatal structural flaws (missing control baseline, abstract mismatch) and the implausible evaluation metrics are not addressable in a rebuttal — they reflect fundamental methodological gaps. The substantive finding is already known. Even if all issues were corrected, the two-character, single-language, small-scale setup would be insufficient to support generalizable conclusions.

*Originality:* Very low — no new method, no new theory, rediscovering an established result.
*Importance of research question:* Moderate in principle (personalized DA for low-resource character dialogue), low in practice (scope too narrow to generalize).
*Claims well supported:* No — the core claim is untested (no unaugmented baseline) and the quantitative metrics are uninterpretable without a described evaluation protocol.
*Soundness of experiments:* Poor — missing critical control, suspicious metric values, no dataset size information.
*Clarity of writing:* Below average — abstract contradicts paper body, Section 4.3 reasoning is partially confused.
*Value to the research community:* Very low in current form.

**Score: 2.0 / 10 — Reject**

MY FINAL SCORE: <pineapple>2.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>