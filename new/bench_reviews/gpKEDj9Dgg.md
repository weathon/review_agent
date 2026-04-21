Now I have sufficient calibration. Let me compile my final review.

## Summary

This paper proposes a two-stage framework for improving medication name transcription in clinical ASR: Whisper-Large-v3 generates N-best hypotheses, which are rescored by a LoRA-adapted LLM to produce corrected transcriptions. The approach is evaluated on a custom dataset (~600 medication names) and reports improved results over fine-tuning the ASR model directly.

## Strengths

- **Well-motivated problem**: Medication name errors in clinical ASR (e.g., "hyper-" vs. "hypo-", "rifampin" vs. "rifampicin") are a genuine patient-safety concern, and the introduction clearly articulates this with concrete examples (Section 1).
- **Architecturally sound concept**: The two-pass design (ASR → LLM rescoring) is a reasonable and well-precedented framework; prior work (Whispering LLaMA, Hyporadise) validates this direction (Section 2).
- **Clear workflow visualization**: Figure 1 effectively illustrates the pipeline including the LoRA decomposition (W_A W_B = ΔW), making the architecture immediately understandable.

## Weaknesses

### Fatal

- **The central claim of "WER reduction" is unsupported — no WER is reported.** The abstract states "a significant reduction in Word Error Rate (WER)," but the evaluation section (Section 4.1) specifies that "We used ROUGE score to evaluate the performance of the model." ROUGE measures n-gram overlap and is a summarization metric, not an ASR evaluation metric. It does not directly measure transcription accuracy, and improvements in ROUGE do not imply reductions in WER — especially for medication names where a single character change (e.g., "rifampin" → "rifampicin") yields a large WER penalty but only a small ROUGE difference. Table 1 reports unlabeled scalars (13.45, 25.10, 7.98, 7.45), which are presumably ROUGE scores, making the headline claim entirely unsupported.

### Major

- **No before/after comparison with the ASR baseline.** Section 4.2 mentions that "finetuning the ASR model itself... achiev[ed] a benchmark of 21%" but this experiment is not described in the methodology, the metric is unspecified (21% WER? 21 ROUGE?), and no results table shows the raw Whisper output vs. the LLM-rescored output. Without reporting Whisper's unrescored output on the test set, there is no way to assess whether LLM rescoring actually helps. A rescoring paper without a before/after comparison has no verifiable results.

- **Model identity contradiction between sections.** The abstract and Figure 1 state the LLM is "LLaMA 3," while Section 4.1 states it is "Llama-2-8b Instruct." These are different architectures with different capabilities. It is unclear which model was actually used, making the results unverifiable.

- **Results are uninterpretable.** Table 1 reports four numbers (13.45, 25.10, 7.98, 7.45) at epochs 7, 9, 11, and 13 with no metric label, no variance, and no explanation for the non-monotonic pattern (the value increases from epoch 7→9 then drops). Four scalar numbers with no error bars, no statistical testing, and no metric identification cannot support any conclusion.

### Minor

- **Dataset and pipeline are underspecified.** The "Pharma-Speak" dataset is described only as "about 600 medication names... separated to about 506 rows for training and the rest for testing" (Section 4.1). It is unclear whether this is speech data, text data, or a name list. The N-best generation, prompt formatting, and rescoring mechanism (scoring vs. generation) are never specified. With ~94 test examples and no variance, generalizability is limited.
- **Hyperparameter choices are unusual and lack justification.** Batch size 64 with ~506 training rows implies ~8 gradient steps per epoch. LoRA rank r=4 is very low. 15 epochs on a ~500-sample dataset raises overfitting concerns. None of these choices are ablated or justified (Section 4.1).
- **The novelty claim is overclaimed.** The introduction states this is "the first of its kind done within the medication name domain" but LLM-based ASR rescoring (Whispering LLaMA, Hyporadise) is extensively studied; applying it to a new lexical subdomain without new methodology does not constitute a first-of-its-kind contribution (Section 1).

### Trivial

- None beyond what is already noted above.

## Nice-to-Haves

- Report WER and entity-level accuracy before and after rescoring to substantiate the headline claim
- Compare against simple baselines like dictionary-based correction or zero-shot GPT correction
- Run the experiment on an existing medical ASR benchmark with standardized evaluation
- Show qualitative examples of audio → Whisper N-best → LLM-corrected output, especially for medication names
- Resolve the LLaMA 2 vs. LLaMA 3 discrepancy and document the prompt template, LoRA target modules, and training objective

## Removed Points

- **Formatting and notation issues**: The harsh critic flags "column headers not explained" and non-monotonic epoch patterns as structural problems. While these are real weaknesses, they are symptoms of the deeper evaluation problems already captured above rather than standalone fatal issues.
- **Missing appendix/proofs**: The parser strips these; not a valid criticism.
- **"Not-yet-released" model availability concerns**: The harsh critic implies the model identity contradiction makes results unverifiable. While the contradiction is real, we do not question whether either model exists — both LLaMA 2 and LLaMA 3 are released. The issue is which one was actually used.
- **Reproducibility nitpicks about undisclosed hyperparameters**: Minor given the paper already lists hyperparameters (batch size, epochs, learning rate, LoRA rank). The more important issue is that the training objective and prompt format are unspecified.
- **The Strength Finder's claim about "demonstrated WER reduction... from 21% to 7.45%"**: This is directly contradicted by the paper itself, which evaluates using ROUGE, not WER. The 21% and 7.45 values are either different metrics or the same metric under different conditions, and conflating them misrepresents the evidence. Moved here as it conflicts with a verified Fatal weakness.

## Novel Insights

The paper's most interesting observation — that LLM rescoring may fix clinically critical medication name errors in ASR — is conceptually sound but remains entirely unvalidated at the metric level. The fundamental tension is that a paper about reducing word-level transcription errors used a summarization metric instead, leaving the core hypothesis untested.

## Suggestions

- **Replace ROUGE with WER as the primary evaluation metric.** This is essential — without WER, the paper's central claim cannot be evaluated.
- **Add a baseline row** showing Whisper-Large-v3's raw WER on the same test set, so readers can assess whether LLM rescoring helps at all.
- **Clarify which LLM was actually used** (LLaMA 2 or LLaMA 3) and report the training objective (next-token prediction on corrected transcriptions? log-likelihood scoring of hypotheses?).

## Calibration Anchors

| Paper | Avg Score | Relation |
|-------|-----------|----------|
| CPLLM (Clinical Prediction with LLMs) | 2.00 | Healthcare+LLM paper with weak experimental design, overclaimed novelty, marginal baselines — similar pattern of applying existing technique without strong validation |
| Explainable medical image clustering | 2.50 | Very limited quantitative evaluation, missing baseline comparisons, unclear methodology — comparable evaluation deficiency |
| Hebrew Speech dataset (aOPTDchLBz) | 2.50 | ASR paper with missing evaluation details (no proper WER on manual transcripts) — shares evaluation deficiency |
| KNN Score for forecasting | 3.00 | Proposed metric doesn't measure what it claims — similar metric mismatch issue |
| Application of Metric Transformation in Retrosynthesis | 2.33 | Metric transformation/mismatch between claimed and actual evaluation — comparable core issue |
| Federated Learning with DP for ASR | 3.00 | ASR paper with reasonable methodology but limited practical conclusions — more competent execution |
| ivrit.ai Hebrew Speech | 2.50 | Dataset/resource paper for ASR, weaker on evaluation but stronger on contribution of new data |

This paper shares the critical flaw of the metric-mismatch papers (avg 2.0–3.0): its headline claim (WER reduction) is unsupported by its actual evaluation (ROUGE). Unlike papers at the 3.0 score, where the methodology was at least internally consistent even if limited, this paper has an internal contradiction between its abstract claim and its experimental instrument. It is somewhat above the weakest anchors (1.5–2.0) because the problem motivation is genuinely important and the architecture is sensible, but below the 3.0 anchors because the evaluation fails to substantiate the core claim.

MY FINAL SCORE: <pineapple>2</pineapple>
MY FINAL DECISION: <orange>Reject</orange>