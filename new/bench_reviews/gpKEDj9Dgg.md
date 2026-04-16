## Summary
This paper proposes a two-stage ASR correction pipeline for medication-name recognition: Whisper-Large-v3 generates N-best hypotheses, and a LoRA-adapted LLaMA model performs second-pass correction/rescoring. The problem is important and clinically meaningful, but the submission’s empirical evidence does not adequately support its central claims about improving ASR in low-resource healthcare settings.

## Strengths
- The paper targets a genuinely important failure mode: incorrect transcription of medication names can have direct clinical consequences, and the motivation is well articulated in the introduction with concrete examples such as confusion between similar drug names.
- The overall pipeline idea is reasonable and aligned with prior ASR error-correction practice: using a strong ASR front-end followed by language-model-based post-correction is a sensible design, and LoRA is an appropriate parameter-efficient adaptation method for constrained settings.
- The paper is scoped around a practically relevant domain specialization rather than a purely toy task, and the authors do acknowledge some limitations of compute and dataset coverage.

## Weaknesses

###: Fatal
- **The evaluation does not convincingly test the paper’s headline claim.** The paper repeatedly claims improved ASR transcription accuracy for “medication-related conversations in healthcare,” including “significant reduction in Word Error Rate (WER)” (Abstract, Conclusion). However, Section 4.1 describes the dataset as “about 600 medication names prescribed globally with their trade names,” split into 506 training rows and the rest for testing, and states: “We used ROUGE score to evaluate the performance of the model.” This is not sufficient to establish improvement on medication-related speech or healthcare conversations. On the face of the paper, the experiment looks much closer to correction over a small medication-name list than end-to-end ASR evaluation on realistic speech.
- **The reported results are not interpretable enough to substantiate the claims.** Table 1 reports only “Epoch” and “Result,” with values 13.45, 25.10, 7.98, and 7.45, but the paper never clearly defines what this “Result” is. This is especially problematic because the abstract claims WER reduction, while Section 4.1 says the evaluation metric is ROUGE. Without a clearly defined metric, the core empirical claim is not verifiable.

### Major:
- **There are serious internal inconsistencies about what model and setup were actually used.** The abstract says the work applied the “LLaMA 3 model,” and Figure 1 labels the LLM as “llama3,” while Section 4.1 states: “The experiment employs the Llama-2-8b Instruct model.” Likewise, the abstract claims WER reduction, but Section 4.1 says ROUGE was used. These are not cosmetic issues; they make it unclear what system was actually trained and evaluated.
- **The paper does not provide a meaningful baseline suite for its central contribution.** If the contribution is that LLM rescoring improves Whisper outputs, the paper needs at minimum a direct comparison against Whisper alone under the same evaluation protocol. Instead, Section 4.2 only states that the result is “significantly better than the finetuning of the ASR model itself with the use of speech dataset achieving a benchmark of 21%,” but this comparison is not properly described, not shown side-by-side, and not tied to a shared, clearly defined metric. This leaves the claimed benefit of the proposed rescoring step unsupported.
- **The methodology is too underspecified to assess the claimed second-pass rescoring mechanism.** Section 3 gives a generic description of LoRA and N-best-to-transcription mapping, but key procedural details are missing: how the N-best hypotheses are formatted, whether the LLM ranks candidates or generates a corrected string, what the prompt is, what the supervision target is, and how inference selects the final output. Since second-pass rescoring is the paper’s core technical claim, this lack of specificity materially weakens the submission.
- **The “low-resource healthcare setting” framing is not empirically established.** The paper motivates the problem using accented, noisy, multilingual, and multi-speaker clinical environments, but the experiment does not actually test such settings. Based on Section 4.1, the evaluation is on a small medication-name dataset, with no direct demonstration on accented speech, noisy clinical audio, multilingual data, or conversational interactions. As written, the evidence supports at most a narrow medication-name correction setting, not the broader low-resource healthcare claim.

### Minor
- **The dataset description is too thin to understand what was actually trained.** Section 4.1 refers to an “open source dataset” of about 600 medication names that the authors curated, but does not clarify whether Pharma-Speak includes audio, text only, ASR outputs, or paired speech/transcript data. This is particularly important because the abstract says Whisper-Large was fine-tuned on a custom dataset, yet the experiment section does not explain that process.
- **Result reliability is hard to assess due to the very small evaluation and limited reporting.** With roughly 94 test rows, and only four epoch values shown out of 15 training epochs, there is no convincing picture of stability, variance, or model selection. I would treat this as secondary relative to the larger evaluation mismatch, but it still weakens confidence in the reported numbers.
- **The novelty is limited.** The paper itself acknowledges that ASR+LLM correction has been “extensively tested in other domains.” Applying second-pass LLM correction to medication names is a reasonable application angle, but the submission does not present a clear methodological innovation beyond domain targeting.

### Trivial
- None.

## Nice-to-Haves
- Include concrete qualitative examples showing Whisper errors on medication names and how the LLM correction changes them.
- Add analysis by medication type, e.g., generic vs. brand names, long vs. short drug names, and phonetically confusable names.
- Report the full learning curve across all epochs rather than four selected checkpoints.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **Concerns about code/dataset availability (“available upon request”).** This is a reproducibility complaint, but under the review instructions this should not be treated as a core weakness by itself. The deeper issue is not artifact release; it is that the experimental setup is under-described in the paper.
- **Complaints about missing hyperparameter details such as beam size, LoRA alpha/dropout, or full implementation minutiae.** These would help reproducibility, but they are not the decisive problems here.
- **Claims doubting the existence/release status of cited models or benchmarks.** Per instruction, these must be removed.
- **Broad requests for many more related-work comparisons by name.** The important point is the lack of meaningful baselines against the paper’s own claimed contribution; I do not rely on unverified external omissions.

## Novel Insights
The central issue is not merely that the paper is “preliminary,” but that its problem formulation drifts between three different tasks without clearly committing to one: (1) low-resource healthcare conversational ASR, (2) medication-name recognition in speech, and (3) small-scale text-level correction over a medication lexicon. The motivation and conclusions are written for (1), the practical examples suggest (2), but the experiment as described looks closest to (3). That task mismatch is what most fundamentally undermines the paper: even if the reported numbers are real improvements, the current evidence would only support a much narrower claim than the title, abstract, and conclusion assert.

## Suggestions
- Rebuild the empirical section around the actual claimed task: run the full ASR → N-best → LLM pipeline on real medication-related speech and report WER as the primary metric.
- Make the paper internally consistent about the model used: if the experiment used Llama-2-8b Instruct, revise the abstract/figure accordingly; if LLaMA 3 was used, revise Section 4.1.
- Clearly describe the dataset and training instances: specify whether Pharma-Speak contains audio, transcripts, N-best hypotheses, or only medication names; explain how Whisper was fine-tuned and how LLM training examples were constructed.
- Add essential baselines under a shared protocol: Whisper 1-best alone, Whisper plus the proposed rescoring, and at least one simple lexicon-based or edit-distance correction baseline.
- Clearly label Table 1 and report the actual metric used; if WER is claimed, WER must be shown.
- Narrow the claims if the available data only supports isolated medication-name correction rather than broader low-resource healthcare ASR.

## Score and Decision
**Originality:** limited; this is largely an application of known ASR+LLM rescoring ideas to a medication-name setting.  
**Importance of research question:** high; the target problem is clinically meaningful.  
**Whether the claims are well supported:** poor; the paper overclaims relative to the described data, metrics, and results.  
**Soundness of experiments:** weak; the task/metric mismatch, absent core baselines, and under-specified setup are major issues.  
**Clarity of writing:** below bar; the motivation is understandable, but the technical and experimental descriptions are inconsistent.  
**Value to the research community:** currently low, because the paper does not yet provide reliable evidence for the claimed contribution.

For calibration, I compared this submission against:
- **“It’s Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition”** (`/home/wg25r/review_agent/human_reviews/QqjFHyQwtF.md`), an accepted ASR+LLM correction paper with solid WER-based evaluation, clear benchmarks, and stronger technical contribution. This submission is well below that bar.
- **“Listening to Formulas”** (`/home/wg25r/review_agent/human_reviews/pflsJ6V6CL.md`), a lower-scored paper with limited novelty but still a substantially more concrete dataset and broader evaluation than the current paper; this suggests the present submission belongs in the low reject range.
- **“Speech Robust Bench”** (`/home/wg25r/review_agent/human_reviews/D0LuQNZfEl.md`), an accepted benchmark paper whose strength comes from detailed setup and extensive evaluation; again, the current paper is far below that standard.
- **“Do Current Large Language Models Master Adequate Clinical Knowledge?”** (`/home/wg25r/review_agent/human_reviews/gYcft1HIaU.md`), a rejected medical LLM paper where claims were judged somewhat overbroad relative to evidence. The current paper has an even more severe mismatch between claims and evidence.

Given those anchors, this paper is substantially weaker than marginal accept papers and fits a clear reject. The decisive reason is not merely modest novelty, but that the current experiments do not adequately validate the paper’s stated contribution.

MY FINAL SCORE: <pineapple>2.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>