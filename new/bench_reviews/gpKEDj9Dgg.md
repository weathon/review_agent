Now let me read some of the calibration papers mentioned by the Human Finder reviewer.I have now read the full paper and calibration materials. Here is the consolidated final review.

---

## Summary

This paper proposes a two-stage pipeline combining Whisper-Large-v3 ASR with LoRA-fine-tuned LLaMA for second-pass N-best rescoring to improve medication name recognition in low-resource healthcare settings. The authors curate a list of ~600 medication names and trade names ("Pharma-Speak"), fine-tune the LLM on this data, and compare results to direct ASR fine-tuning. The paper is motivated by genuine patient-safety concerns around medication-name transcription errors.

---

## Strengths

- **Clinically meaningful problem**: Medication name misrecognition (e.g., "hyper-" vs. "hypo-," "rifampin" vs. "rifampicin") has direct patient-safety consequences. The motivation is sound and well-articulated.
- **Sensible architectural direction**: The two-pass N-best rescoring paradigm—letting ASR generate candidates and an LLM correct them—is technically grounded and consistent with established approaches (Hyporadise, Whispering LLaMa).
- **Practical adaptation strategy**: Using LoRA avoids full-parameter fine-tuning, which is reasonable given compute constraints; this is a defensible design choice for resource-limited settings.

---

## Weaknesses

### Fatal

**The central empirical claim is not demonstrated by the reported experiment.**

The abstract states "a significant reduction in Word Error Rate (WER)" (directly quoted), but Section 4.1(5) states plainly: "We used ROUGE score to evaluate the performance of the model." Table 1 labels its only column "Result," reporting four numbers (13.45, 25.10, 7.98, 7.45) with no unit or direction of improvement. WER and ROUGE are fundamentally different measures; claiming WER reduction while reporting only ROUGE is not a framing issue—it is an unsupported core claim. The headline result of the paper is therefore unverifiable.

**The experiment does not involve actual speech data.**

The dataset described in Section 4.1(4) is "an open source dataset which had about 600 medication names prescribed globally with their trade names which we curated ourselves." This is a text list, not a speech corpus. The paper provides no description of how audio was recorded, whether any human speakers were involved, how Whisper-Large-v3 was applied to produce N-best hypotheses (from real audio, TTS, or corrupted text), or what the ground-truth transcripts correspond to. For a paper whose entire contribution is framed as ASR rescoring in low-resource healthcare speech, the absence of any speech data in the reported experiment breaks the evidentiary chain entirely. The abstract separately refers to fine-tuning "Whisper-Large ASR model on a custom dataset, Pharma-Speak," but Pharma-Speak is never defined or described beyond a brief mention. No speech collection methodology appears anywhere in the paper.

### Major

- **LLaMA 2 vs. LLaMA 3 contradiction**: The abstract says "applied the LLaMA 3 model," Figure 1 labels the LLM "llama3," but Section 4.1(1) states "The experiment employs the Llama-2-8b Instruct model." These are distinct model families. Readers cannot determine which model was actually used, which undermines any comparison or reproducibility.

- **Table 1 is uninterpretable**: Only 4 of 15 training epochs (7, 9, 11, 13) are reported with no explanation for the selection. The metric is unnamed. Performance worsens dramatically at epoch 9 (25.10 vs. 13.45 and 7.98) with zero discussion of why. The non-monotonic behavior could indicate instability, overfitting, or a reporting error—none of these possibilities is addressed.

- **Baseline comparison is meaningless**: Section 4.2 states the result is "significantly better than the finetuning of the ASR model itself with the use of speech dataset achieving a benchmark of 21%." This single sentence provides no information about what dataset was used, what split, what metric (21% of what?), whether the same test set was involved, or whether this is WER or something else. Without a matched baseline on the same data and metric, the claimed superiority of LLM rescoring over ASR fine-tuning is not established.

- **Missing core methodological details**: The paper gives no prompt template, no description of how N-best hypotheses are formatted as LLM input, no training objective or loss function, no decoding strategy, and no description of how the final transcription is selected from LLM output. These are the operational core of the contribution, and their absence makes the experiment irreproducible.

- **Dataset is too small and lacks speech**: 506 training / ~94 test text entries of medication names is insufficient for fine-tuning an 8B LLM and constitutes a narrow evaluation. There is no acoustic variability, no speaker variability, and no noisy conditions—directly contradicting the paper's framing around accented and noisy low-resource speech settings.

### Minor

- The paper's "first of its kind" novelty claim for the medication domain is plausible as a domain application, but the core method (N-best LLM rescoring with LoRA) is explicitly drawn from prior work (Hyporadise, Whispering LLaMa). The paper does not clearly articulate what is technically new beyond domain focus.

- The conclusion states the approach "significantly enhances the recognition of medication names, even in low-resource environments," but this is overstated given the experimental evidence: no speech data, no WER measurement, no ablation, no statistical test.

### Trivial

- The paper is extremely short (~4 content pages) with thin methodology and results sections. While length alone is not a criterion, the brevity reflects the sparseness of substance.

---

## Nice-to-Haves

- Ablation over LoRA rank (r=4 is very low for an 8B model in 8-bit quantization; r=8, 16, 32 are common) to understand whether the bottleneck is the adapter capacity.
- A learning curve across all 15 epochs (not just 4 selected ones) with a validation curve to explain epoch selection.
- Qualitative examples of N-best rescoring: what Whisper produced, what the LLM output, and what the ground truth was.
- Error analysis categorizing the types of medication errors the LLM fixes vs. fails on (e.g., brand vs. generic, phonetically similar drugs).
- Comparison with simple post-processing alternatives (dictionary lookup, string edit-distance correction) as a sanity-check baseline.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Reproducibility concern (code "available upon request")**: Removed per hard rules on reproducibility nitpicks. This does not affect evaluation of the method itself.
- **LoRA rank r=4 not justified as a weakness**: Moved to Nice-to-Haves; this is a design choice that is underexplored but not fatal.
- **Requesting confidence intervals / statistical tests**: Moved to Nice-to-Haves; with ~94 test samples single-run evaluation may be norm, though given the tiny dataset it would be useful.

---

## Novel Insights

None beyond the paper's own contributions. All three reviewers independently converged on the same fundamental observation—that the experiment does not actually instantiate the ASR pipeline the paper describes—which itself is a clear signal of a foundational gap rather than an area for incremental improvement.

---

## Suggestions

1. **Collect or use real speech data**: The paper cannot be an ASR paper without audio. Use AfriSpeech-200 (cited by the authors' own references, Olatunji et al., 2023) or similar clinical-domain speech corpora to build a real evaluation.
2. **Report a single consistent metric throughout**: If WER is the claim, measure WER. If ROUGE is the evaluation, remove WER claims from the abstract and conclusion. Table 1 must label its metric column.
3. **Resolve the LLaMA 2/3 discrepancy**: Decide which model was used, report it consistently in all sections and figures, and justify the choice.
4. **Include a Whisper-alone baseline**: Without knowing baseline Whisper WER on the test set, no claim about rescoring improvement can be made.
5. **Describe the full data pipeline**: How was audio generated or collected? How did Whisper produce N-best lists for the medication name list? What prompt format was used? These must be specified.
6. **Report all 15 epoch results** with a validation curve to justify model selection.

---

## Score and Decision

**Calibration:**

| Paper | Topic | Issues | Human Score | Decision |
|---|---|---|---|---|
| QqjFHyQwtF | ASR+LLM rescoring (UADF) | Solid method, real experiments, minor framing | 10, 6, 6, 6, 5 | Accept |
| sjvz40tazX | Low-resource ASR benchmark | Some baseline gaps but real data and tasks | 8, 3, 5, 6 | Reject |
| g7DHM6MRE4 | Medical MT (Luganda), LoRA | No technical novelty, some methodology issues, but coherent real experiments | 3, 5, 3, 3 | Reject |
| ztpy1gsUpT | Medical SLM, healthcare | Limited novelty, real experiments, coherent claims | 6, 6, 6, 6 | Accept |

The paper under review is substantively weaker than **g7DHM6MRE4** (human scores 3,5,3,3), which at least had real experiments, an interpretable metric, and coherent claims even if methodologically imperfect. This paper's core deficiency—that the experiment does not involve speech data at all and contradicts its own metric claims—is more fundamental than anything in those rejected papers. The paper is closer to a preliminary project report than a conference submission. Even as a position paper or extended abstract, the contradictions in metrics, models, and data would need to be resolved before the work could be evaluated meaningfully.

**Originality**: Very low — core method is directly drawn from Hyporadise and Whispering LLaMa; the domain application is the only potential novelty.
**Importance of research question**: Moderate to high — medication-name ASR is clinically relevant.
**Claims well-supported**: No — the headline WER claim is contradicted by the evaluation section; the experiment does not use speech data.
**Soundness of experiments**: Very poor — no speech corpus, uninterpretable table, no baselines, contradictory model description.
**Clarity of writing**: Poor — key definitions ("Pharma-Speak," the actual metric, the model used) are inconsistent or missing.
**Value to research community**: Minimal in current form.

**Final Score: 2.0 / 10 — Reject**

MY FINAL SCORE: <pineapple>2.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>