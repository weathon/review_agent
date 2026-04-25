Now let me search for calibration papers.Now I have sufficient information to write the final review. Let me synthesize everything.

---

## Summary
Delta is a training-free, inference-time contrastive decoding method for LLMs that mitigates hallucinations by randomly masking input tokens, running a second forward pass on the masked input, and subtracting the resulting logits (hypothesized to be hallucination-amplified) from the original logits. The method draws inspiration from Visual Contrastive Decoding (VCD) for vision-language models and is evaluated on six QA benchmarks using a single model (Llama 3.1 8B Instruct, 4-bit quantized).

---

## Strengths

- **Large improvement on no-answer detection (SQuAD v2):** Delta achieves a 14.53 pp gain on NoAns_EM (Table 1, 23.63 → 38.17 without sampling), showing a genuine and interpretable benefit in preventing hallucinated answers when no valid answer exists in context.
- **Training-free, inference-time deployment:** The method requires only two forward passes with no parameter updates (Equations 3 and 5, Sections 3.4–3.6), making it deployable on any existing model.
- **Robustness to hyperparameter choices:** The ablation over masking ratios (0.3, 0.5, 0.7) and α (0.1–0.5) on SQuAD v1.1 shows standard deviations of only 0.66 (EM) and 0.21 (F1), and all configurations exceed the baseline (Figure 2, Section 6).
- **Honest reporting of negative results:** The paper explicitly reports marginal declines on CommonsenseQA (−0.25 pp) and MMLU (−0.29 pp) (Table 2, Section 5.3), and the without-sampling degradation on TriviaQA and NQ is included in Table 1 rather than omitted.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing comparison with Context-Aware Decoding (CAD):** Section 2 explicitly discusses CAD (Shi et al., 2024) as a method that "amplifies differences between outputs generated with and without the given context." This is nearly identical in mechanism to Delta (which amplifies differences between full vs. masked-context outputs). The paper claims Delta is "more generalizable" than CAD because CAD is "mainly based on context-driven datasets," yet Delta itself shows no improvement on context-free benchmarks (Table 2). That claimed distinction provides no actual differentiation. Without a head-to-head comparison against CAD on the same datasets and model, the paper cannot demonstrate that its contribution advances the state of the art.

- **Single-model evaluation undermines generality claims:** All experiments use exactly one model — Llama 3.1 8B Instruct with 4-bit quantization. The abstract and conclusion claim Delta is "computationally efficient and scalable" and "a powerful solution for real-world LLM applications," but these claims are unsupported by a single-model study. Whether gains transfer to other model families (Mistral, Qwen, Gemma), other scales (3B, 70B), or full-precision settings is entirely unknown. The interaction between 4-bit quantization and the logit-subtraction procedure is also unexplored.

- **DoLa is cited but never compared:** DoLa (Chuang et al., 2024) is introduced in the very first paragraph of the introduction as a contrastive decoding baseline, but never appears in any results table. DoLa is training-free and inference-time, the same category as Delta, making its omission unjustified.

### Minor

- **Unvalidated mechanistic hypothesis:** Section 3.2 claims that masking input tokens *specifically* amplifies hallucinations in the resulting logits. This is illustrated by one constructed example (the banana sentence) but never empirically validated. The plausible alternative — that masking simply degrades prediction quality uniformly, and the contrastive formula works because it sharpens the original distribution (similar to temperature reduction) — is not ruled out. This does not invalidate the empirical results but does leave the theoretical motivation of the paper on weak footing.

- **HasAns/NoAns tradeoff in SQuAD v2 unanalyzed:** The large NoAns_EM gain (+14.53 pp) is accompanied by a HasAns_EM *decline* (59.08 → 57.47 without sampling, Table 1). Delta is shifting the model toward abstention. This precision/recall tradeoff is not analyzed and the overall 6 pp EM gain masks a redistribution of errors rather than a uniform improvement.

- **EOS token used as MASK token with no justification:** Section 4.2 states that "all experiments utilize the end-of-sequence (eos) token as the MASK token." This is an unusual and potentially problematic choice — an instruction-tuned model may interpret EOS-replacing tokens as sequence termination signals, producing erratic behavior. No ablation or justification is provided.

- **Ablation study limited to SQuAD v1.1:** The ablation covers only the dataset where Delta performs most consistently. Ablations on TriviaQA or NQ (where behavior is weaker or condition-dependent) are absent. Additionally, α=0 is not tested as a control, making it impossible to separate the contributions of the contrastive decoding component from the Adaptive Plausibility Constraints (APC) filter alone.

### Trivial
None.

---

## Nice-to-Haves
- An ablation comparing EOS-as-MASK against a learned `[MASK]` token or a zero-embedding to justify the implementation choice.
- Qualitative error analysis on TriviaQA without sampling, where Delta slightly hurts performance, to reveal what types of errors the method introduces.
- Extension to at least one additional model family to provide minimal evidence of generalizability.

---

## Removed Points
*These points are flagged to be removed; treat them with caution.*

- **"ICD misattributed to Leng et al."** (Harsh Critic): The paper assigns both VCD and ICD to "Leng et al. (2024)" in Section 2. However, per the hard rules, we cannot confirm external citation details, and this is at most a minor bibliographic issue, not a substantive criticism of the method.
- **Concern about whether datasets/models exist or are available:** Any implicit concern about the availability of Llama 3.1, SQuAD v2, TriviaQA, NQ, etc. is removed per the hard rule — if the paper cites them, they exist.
- **Request for statistical significance tests / confidence intervals** (Harsh Critic): Moved to nice-to-have. Single-run evaluation is the norm at this scale; demanding CIs for every table entry is not standard practice in this community.
- **Generic strength: "training-free approach is important"** (Strength Finder): Removed as generic — this applies to any inference-time method.
- **Generic strength: "addresses an important problem"** (Strength Finder): Removed as generic.

---

## Novel Insights
The paper's most genuinely interesting finding is the asymmetric effect of Delta on SQuAD v2: it substantially improves no-answer detection at the cost of a small drop in has-answer extraction. This suggests that contrastive decoding with masked inputs effectively lowers the model's confidence threshold for committing to a specific answer, which is beneficial when the correct answer is "no answer" but harmful when an answer genuinely exists. This precision/abstention tradeoff is a concrete, interpretable observation that would deserve deeper analysis in future work.

---

## Evaluation on Key Axes

- **Originality:** Low-to-moderate. The core idea is a direct adaptation of VCD (Leng et al., 2024) to text via token masking. The translation is principled but not novel in mechanism.
- **Importance of research question:** High. Hallucination mitigation in LLMs is a central open problem.
- **Claims vs. support:** Poor. Central claims of scalability and generality rest on a single model experiment. The claim of superiority over CAD is asserted but not tested.
- **Soundness of experiments:** Weak. Single model, single quantization setting, missing the closest baseline (CAD), missing DoLa.
- **Clarity of writing:** Acceptable. The method is clearly described and tables are readable.
- **Value to research community:** Limited in current form. The SQuAD v2 no-answer result is a genuine finding, but insufficient to establish the method's place in the landscape without CAD and DoLa comparisons.

---

## Calibration Anchors

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| DoLa (contrastive decoding, multi-model) | `Th6NyL07na.md` | **7.25** | Much stronger: multi-model, theoretically grounded, strong baselines; Delta falls far short |
| Instructive Decoding (contrastive logit adjustment) | `LebzzClHYw.md` | **7.50** | Accepted spotlight: robust multi-model experiments; Delta has single model |
| Prior-Aware Decoding (contrastive inference, rejected) | `dlUjNdybnq.md` | **5.50** | Rejected but tested 11 models across 44 task-model combinations; Delta evaluates 1 model |
| Thought-Retriever (single-model, withdrawn) | `SkDNQbMQba.md` | **3.75** | Single-model with limited scope, withdrawn; comparable level of evaluation to Delta |
| Missing-baselines paper (score ≤ 4) | `fmHS8aBfuH.md` | **3.00** | Rejected for missing baselines and weak empirical evidence; Delta has same issues |

Delta's combination of single-model evaluation, missing the closest baseline (CAD), and missing an obvious comparison (DoLa) places it below the rejected PAD paper (5.5, which tested 44 model-task combinations) and closer to the withdrawn/rejected papers in the 3–4 range. The genuine positive results on SQuAD prevent a score as low as 2.5–3.0. A score of **3.5** reflects a method with a real signal (SQuAD v2 no-answer) but insufficient experimental rigor to establish its contribution.

---

## Score and Decision

The paper presents a reasonable idea with some positive empirical results, but has two decisive gaps that prevent acceptance: no comparison against CAD (the near-identical prior method it explicitly discusses) and single-model evaluation that cannot support any generality claim. Relative to calibration anchors, the rejected PAD paper (5.5) tested 11 models and 44 combinations; the withdrawn Thought-Retriever (3.75) and fmHS8aBfuH (3.0) share similar baseline-gaps and limited scope. Delta's genuine SQuAD v2 finding keeps it above the weakest cluster (2.5–3.0), but the structural gaps in evaluation place it squarely at the low end.

MY FINAL SCORE: <pineapple>3.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>