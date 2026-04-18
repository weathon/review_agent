## Summary

This paper proposes a two-stage framework for improving medication name recognition in ASR for low-resource healthcare settings: Whisper-Large-v3 generates N-best hypotheses, which are then rescored by an LLM fine-tuned with LoRA. The authors curate a Pharma-Speak dataset of ~600 medication names and report results showing improvement over a 21% ASR-only fine-tuning baseline.

## Strengths

- **Important and practical problem**: Medication name transcription errors in clinical settings carry real patient safety risks, and even strong general ASR models struggle with domain-specific terminology. This is a well-motivated application.
- **Conceptually sound pipeline direction**: The two-pass architecture (ASR → LLM rescoring) with LoRA for parameter-efficient adaptation is a reasonable and established paradigm, and applying it to medication name correction is a valid niche.
- **Acknowledges limitations**: The paper explicitly notes GPU constraints, incomplete drug coverage, and the gap between brand vs. chemical names (Section 4.3).

## Weaknesses

### Fatal

- **The claimed WER improvements are not supported by the reported experiments.** The abstract states "a significant reduction in Word Error Rate (WER)" and the conclusion claims "significantly enhances the recognition of medication names," yet Section 4.1 states "We used ROUGE score to evaluate the performance of the model." Table 1 reports a single "Result" column (13.45, 25.10, 7.98, 7.45 at epochs 7, 9, 11, 13) without specifying whether these are WER, ROUGE-1, ROUGE-L, or another metric. The comparison to a "benchmark of 21%" for ASR fine-tuning is almost certainly in WER, making any comparison with the unlabeled Table 1 values incoherent. Without consistent, clearly defined metrics and a proper apples-to-apples comparison, the paper's central empirical claims are unsubstantiated.

- **No baseline comparison demonstrating that LLM rescoring actually improves over Whisper's own outputs.** Table 1 only shows LLM fine-tuning performance across epochs; there is no comparison between the raw Whisper N-best best-1 output and the LLM-rescored output on the same test set with the same metric. There is also no oracle N-best analysis to show whether the correct transcription is even present in the N-best list. The single baseline ("21%") is vaguely attributed to "finetuning of the ASR model itself" without specifying the metric, test set, or conditions, and it is compared against Table 1 values of unclear metric identity. This means the paper cannot establish that the proposed method improves upon the ASR baseline at all.

### Major

- **The model identity is inconsistent across the paper.** The abstract and Figure 1 reference "LLaMA 3," Section 4.1 specifies "Llama-2-8b Instruct," and Section 4.4 discusses "LLaMa 3.1" as future work. These are fundamentally different models (different architectures, sizes, and capabilities). It is unclear which model was actually used, which undermines both reproducibility and the credibility of the reported results.

- **The dataset is extremely small and critically underspecified.** The "Pharma-Speak" dataset is described as "about 600 medication names prescribed globally with their trade names which we curated ourselves," split into ~506 training and ~94 test rows. Crucially, the paper never clarifies whether this is audio data or a text-only list of names. If it is text-only, it is unclear how Whisper generates N-best hypotheses from it. If it includes audio, no information is given about speakers, accents, noise conditions, or recording setup—despite the introduction emphasizing the challenges of accented, noisy clinical environments. A dataset of ~94 test items is far too small to support the paper's generalization claims.

- **The methodology is described at too high a level to be reproducible or assessable.** The paper never specifies: (a) how N-best hypotheses are formatted as LLM input, (b) what prompt template is used, (c) whether the LLM selects from the N-best list or generates a corrected transcription, (d) the training objective (seq2seq generation? classification? ranking loss?), or (e) how N-best lists are obtained from Whisper (beam size, decoding parameters). Without these details, it is impossible to distinguish the method from standard text correction or assess its novelty.

- **Selective and unstable training results.** Table 1 reports results only at epochs 7, 9, 11, and 13 out of 15 total epochs, without explanation for these specific choices. The result at epoch 9 (25.10) is dramatically worse than epoch 7 (13.45), indicating training instability, which is unacknowledged and unexplained. Showing only 4 scattered epoch points without the full training curve raises concerns about selective reporting.

### Minor

- **ROUGE is a questionable metric for medication name correction.** Medication names are short, typically single-word or few-word sequences. ROUGE (a recall-oriented metric designed for summarization) is poorly suited for this task; exact-match accuracy, character error rate, or named entity-level F1 would be more appropriate and interpretable.

- **The novelty claim is overstated.** The paper states this is "the first of its kind done within the medication name domain," but the method (LLM + LoRA for N-best rescoring) is essentially the same pipeline as Whispering LLaMA (Radhakrishnan et al., 2023) and HyPoradise (Chen et al., 2024), both of which the paper cites. Domain-specific application alone, with no technical innovation and no convincing evaluation, provides limited novelty.

- **Mismatch between motivation and experiments.** The introduction extensively discusses accented speech, noisy clinical environments, and multi-speaker dialogues, but the experiments provide no evidence that the data reflects any of these conditions.

### Trivial

- The abstract mentions fine-tuning Whisper-Large on Pharma-Speak, but Section 4.1 only describes LLM fine-tuning, making it unclear whether Whisper was also fine-tuned and on what data.

## Nice-to-Haves

- Report WER and entity-level accuracy on the same test set for both Whisper baseline (1-best and oracle N-best) and the LLM-rescored output to enable direct comparison.
- Include qualitative examples showing N-best hypotheses before and after LLM rescoring to illustrate the types of errors corrected and introduced.
- Evaluate on a standard healthcare ASR benchmark (e.g., a subset of AfriSpeech) with proper WER reporting to allow comparison with prior work.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Reproducibility concerns about "code and dataset available upon request"**: Per rules, reproducibility nitpicks about implementation details and artifact availability are removed from weaknesses. The dataset being available upon request is a minor style issue, not a core flaw.
- **Demand for confidence intervals on the small test set**: While this would strengthen the paper, single-run evaluation is the norm in this research community for small-scale experiments. This is a nice-to-have, not a core flaw.
- **Missing comparison with domain-specific LLMs (BioBERT, MEDITRON)**: This is scope creep. The paper's stated method uses LLaMA with LoRA; demanding additional models goes beyond the paper's scope.

## Novel Insights

The paper surfaces a genuine and underexplored problem—medication name errors in ASR for low-resource clinical settings—but its execution is too preliminary to provide meaningful insights on how LLM rescoring performs in this domain. The inconsistent metric reporting and absence of proper baselines make it impossible to draw conclusions about effectiveness from the current results.

## Suggestions

1. **Resolve the metric confusion immediately**: Report both WER and ROUGE for all experimental conditions, clearly label every number in every table, and ensure all comparisons use the same metric on the same test set.
2. **Establish proper baselines**: At minimum, report (a) Whisper 1-best WER on the test set, (b) Whisper oracle N-best WER, and (c) LLM-rescored WER, all on the same data with the same metric.
3. **Clarify the dataset**: State explicitly whether audio data exists, how it was collected, and provide basic characterization (number of speakers, accents, noise conditions). If no audio data exists and the experiment is text-only, be transparent about this and revise claims about "speech recognition" improvement.
4. **Report the full training curve**: Show all 15 epochs and explain the instability between epochs 7–9.
5. **Specify the exact model used**: Resolve the LLaMA 3 vs. Llama-2-8b inconsistency.

## Score and Decision

Calibration against similar papers:

- **ceATjGPTUD** (LLM+ASR GER, strong evaluation, accepted spotlight): avg ~8. This paper shows what proper LLM+ASR work looks like—clear metrics, full baselines, ablations.
- **n4SLaq5GhM** (healthcare NLP, no empirical validation): avg ~3.25. Rejected. Similar pattern of claims without supporting evaluation.
- **pflsJ6V6CL** (speech recognition, dataset concerns): avg ~4. Rejected/withdrawn.
- **Gh1XW314zF** (multimodal healthcare, limited evaluation): avg ~3.5. Rejected/withdrawn.
- **1WSd408I9M** (generative AI healthcare, no proper evaluation): avg 1. Rejected.

This paper identifies a meaningful problem and follows a reasonable architectural direction, but the experimental execution has fundamental flaws: the primary metric (WER) claimed in the abstract is not actually reported; the evaluation metric used (ROUGE) is mismatched with the baseline comparison (likely WER); there is no direct baseline comparison demonstrating improvement; the model identity is inconsistent; the dataset is tiny and unspecified; and the methodology lacks critical detail. These are not fixable in a rebuttal—they require a fundamentally redesigned and re-run evaluation. The paper is closer in quality to the rejected healthcare NLP papers (scores 3–4) than to the accepted ASR+LLM papers.

MY FINAL SCORE: <pineapple>2.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>