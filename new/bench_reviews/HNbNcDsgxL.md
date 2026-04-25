## Summary

Delta proposes a contrastive decoding method that randomly masks input tokens and subtracts the masked logits from the original logits to mitigate hallucinations in LLMs. It demonstrates improved QA accuracy on context-rich benchmarks—especially a 14.53 percentage point gain on SQuAD v2 no‑answer exact match—but the evaluation relies on answer‑accuracy metrics rather than direct hallucination measures.

## Strengths

- **Simple, inference‑time intervention**: Delta requires no model retraining and uses fixed hyperparameters, making it computationally efficient and easy to deploy. (Section 4.2: “parameters fixed at \(r_{\text{mask}} = 0.7\), \(\alpha = 0.3\), and \(\beta = 0.1\)”)
- **Consistent gains across multiple QA datasets**: Table 1 shows improvements of 3–7 percentage points on SQuAD v1.1, SQuAD v2, TriviaQA (with sampling), and Natural Questions (with sampling).
- **Strong performance on unanswerable questions**: The 14.53 pp gain on SQuAD v2 no‑answer EM (non‑sampling) and 11.81 pp (sampling) indicates the method helps models refrain from answering when context does not support an answer. (Section 5.1)
- **Robust hyperparameter sensitivity**: The ablation in Section 6/Figure 2 shows Delta outperforms the baseline across a wide range of mask ratios (0.3–0.7) and temperatures, with low standard deviation (0.66 EM, 0.21 F1).
- **Clear intuitive example**: Figure 1’s “moldy banana” illustration effectively demonstrates how masked inputs trigger prior‑based guesses and how contrastive subtraction corrects this.
- **Honest limitation acknowledgment**: Section 5.3 correctly identifies that Delta has “marginal effectiveness” on context‑free tasks (CommonsenseQA, MMLU), demonstrating scientific awareness of scope.

## Weaknesses

### Major

1. **Core claim unsupported by evaluation metrics** – The paper’s central contribution is “hallucination mitigation” (title, abstract, Section 3), yet evaluation uses only Exact Match and F1 scores on QA datasets. These measure answer accuracy, not hallucination reduction directly. While SQuAD v2 no‑answer improvements suggest better abstention, they do not demonstrate reduced fabrication in generated answers; no hallucination‑specific benchmark (e.g., TruthfulQA, HaluEval) or factuality metric is employed. The evidence therefore does not support the broad claim.

2. **Missing comparisons to relevant hallucination baselines** – The paper cites CAD (Shi et al., 2024), DoLa (Chuang et al., 2024), and self‑reflection methods but does not compare against them experimentally. Only a vanilla decoding baseline is used. Given that CAD is a direct contrastive competitor, this omission prevents assessing Delta’s relative effectiveness as a hallucination mitigation technique.

3. **Unverified foundational hypothesis** – The method rests on the assumption that “masking portions of input text … can exacerbate hallucinations” (Section 3.2). This is illustrated with an example but never empirically validated. No experiment compares hallucination rates or token distributions between masked and unmasked prompts. If masked inputs do not systematically increase hallucinated outputs, the contrastive subtraction in Equation (3) lacks theoretical grounding.

4. **Alternative explanation: improved abstention, not reduced hallucination** – The large gain on SQuAD v2 no‑answer EM could stem from the model learning to say “no answer” more often via the decoding constraint, rather than from reducing false content in answered questions. HasAns_EM remains largely unchanged (Table 1: baseline 59.08 → Delta 57.47 w/o sampling; 58.22 → 58.62 w/ sampling), suggesting the primary effect is a shift in answer‑selection threshold. Without analyzing hallucination rates among answered questions, this alternative interpretation remains plausible.

### Minor

1. **No ablation on masking strategy** – The paper uses random masking throughout and lists “advanced masking strategies” as future work (Section 7), but does not test whether random masking is necessary or optimal. Comparing random masking to targeted masking (e.g., content words only) or to a no‑masking condition would help isolate the contribution of the masking component.

2. **Limited analysis on context‑free task failure** – Delta shows negligible change on CommonsenseQA and MMLU (Table 2), which the paper notes as a limitation. However, there is no investigation into *why* masking fails to help—e.g., does it remove useful cues, or does the contrastive signal become noise? Understanding this boundary would clarify the method’s scope and guide appropriate use.

### Trivial
- None substantive beyond minor notation or phrasing improvements.

## Nice‑to‑Haves

- Direct hallucination evaluation on TruthfulQA, HaluEval, or a factuality metric (e.g., QAFactEval).
- Head‑to‑head comparison with CAD (Shi et al., 2024) and DoLa (Chuang et al., 2024).
- Case studies showing before/after generations on hallucination‑prone examples (e.g., SQuAD v2 unanswerable questions).
- Probability distribution visualizations for selected tokens to illustrate contrastive filtering.
- Error analysis on answered questions to distinguish abstention from factuality improvement.

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Equation 4 (Adaptive Plausibility Constraints) is never used in reported experiments** – Actually, APC is incorporated via \(\beta = 0.1\) in all experiments (Section 4.2); it is part of the method though not ablated. This is a minor presentation choice, not a flaw.
- **Future work acknowledges need for “advanced masking strategies,” admitting current random masking is weak** – Acknowledging limitations is good scientific practice; it does not undermine the current contribution.
- **Parser‑induced formatting/typo issues** – Excluded per rules; not author errors.

## Novel Insights

The idea of using intentionally degraded (masked) inputs to approximate hallucination‑prone reasoning and then subtracting those logits is conceptually aligned with contrastive decoding literature. The most concrete demonstrated effect is improved abstention on unanswerable questions, suggesting Delta helps models avoid committing to false answers when context lacks support. However, the claim of general hallucination mitigation overreaches the evidence; the method’s benefit appears specific to contrastive reasoning over context, not to improving factual knowledge or reducing false content in open‑ended generation.

## Suggestions

1. To support the hallucination claim, run experiments on a direct hallucination benchmark (e.g., TruthfulQA) and/or compute factuality metrics (e.g., FEVER score) over generated answers.
2. Compare against CAD and at least one other contrastive or self‑reflection baseline to position Delta relative to prior art.
3. Provide an analysis of hallucination rates *among answered questions* (e.g., precision/recall trade‑off) to disentangle abstention from factuality improvement.
4. Validate the core hypothesis by measuring the factuality or token‑probability divergence between masked and unmasked outputs on a held‑out set.

## Score and Decision

I calibrated against multiple anchor papers:

- **High‑scoring hallucination work**: DoLa (avg 7.25) and the self‑contradiction paper (avg 6.0) both evaluated directly on factuality/hallucination benchmarks (TruthfulQA or custom datasets), compared to relevant baselines, and released code. Delta does not meet these standards.
- **Medium‑scoring contrastive decoding**: PAD (avg 5.5, rejected) lacked strong baselines and relied on synthetic tasks; its theoretical framework was also questioned. Delta shares the baseline gap and adds misaligned evaluation and an unverified core hypothesis, placing it *below* PAD.
- **Low‑scoring contrastive papers** (avg 3–4) suffered from no methodological novelty or poor practical utility. Delta has a clear method and solid QA numbers but fails to evaluate its claimed effect, keeping it in the low‑medium range.

Given the fundamental disconnect between the hallucination‑mitigation claim and the QA‑accuracy evidence, plus the absence of key comparisons and verification of the core hypothesis, the paper does not meet the acceptance threshold.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>