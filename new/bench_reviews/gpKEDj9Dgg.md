Now I have enough calibration data. Let me synthesize the final review.

**Calibration Summary:**

| Paper | Key Issues | Scores | Decision |
|-------|-----------|--------|----------|
| CPLLM | Minimal novelty (LLM+LoRA applied to medical), weak baselines | 1, 1, 3, 3 | Reject |
| Luganda MT | Low-resource domain, applying existing techniques, synthetic dataset quality | 3, 5, 3, 3 | Reject |
| SLM Exploration | Lack of novelty, limited evaluation | 3, 3, 3, 3 | Reject |
| UADF (ASR+LLM) | Proper GER work with novel fusion method, good evaluation | 10, 6, 6, 6, 5 | Accept |
| RobustGER | Novel GER method, thorough evaluation | 8, 6, 10, 8 | Accept |
| Node-CwR | Limited technical contribution, close to existing methods | 3, 3, 3, 3 | Reject |

The current paper falls well below the accepted GER/ASR+LLM papers in terms of novelty and evaluation rigor. It's more comparable to CPLLM (1,1,3,3) and the SLM/Node-CwR papers (3,3,3,3) in terms of applying existing techniques without meaningful adaptation, combined with serious evaluation problems.

Let me now write the final review.

## Summary

This paper proposes a two-stage framework for improving ASR transcription of medication names in low-resource healthcare settings: Whisper-Large-v3 generates N-best hypotheses, and an LLM (LLaMA-2 with LoRA) performs second-pass rescoring to correct errors. The authors curate a custom dataset called Pharma-Speak (approximately 600 medication names), fine-tune the LLM with LoRA, and evaluate using ROUGE scores across selected training epochs.

## Strengths

- **Well-motivated problem domain**: Medication-name transcription errors in clinical ASR can endanger patient safety (e.g., "hyper-" vs. "hypo-," "rifampin" vs. "rifampicin"), making this an important and underserved application area.
- **Reasonable high-level approach**: The two-stage rescoring paradigm (ASR → N-best → LLM correction) is a well-established and practical framework, and LoRA-based adaptation preserves the LLM's general language knowledge while targeting domain-specific correction.

## Weaknesses

### Major

- **Fundamental metric incoherence — claims WER reduction but reports ROUGE**: The abstract claims "significant reduction in Word Error Rate (WER)" and the conclusion states the method "significantly enhances the recognition of medication names." However, Section 4.1 explicitly states "We used ROUGE score to evaluate the performance of the model," and Table 1 reports only unlabeled scalar values (13.45, 25.10, 7.98, 7.45) with no metric name. ROUGE is a summarization/n-gram overlap metric ill-suited for evaluating medication name transcription — for short strings like drug names, small character differences that flip clinical meaning (rifampin vs. rifampicin) may be masked by n-gram overlap. The sole comparative statement ("significantly better than the finetuning of the ASR model itself… achieving a benchmark of 21%") compares what appears to be a WER figure against ROUGE scores, making the comparison meaningless. This undermines the core empirical claim of the paper.

- **Inadequate experimental evidence**: Table 1 contains only 4 data points across 15 training epochs, with no variance, no error bars, and no explanation for the erratic non-monotonic behavior (e.g., the jump from 13.45 at epoch 7 to 25.10 at epoch 9). No baseline comparisons in the same metric appear — there is no side-by-side table showing Whisper-only WER vs. Whisper+LLM WER under identical conditions. No ablation studies are presented (e.g., varying LoRA rank, removing N-best input, varying N). With ~94 test examples and no reported variance, even apparent improvements could be random fluctuation.

- **Minimal novelty beyond applying existing methods**: The approach of using an LLM for N-best rescoring (generative error correction) in ASR is well-established (e.g., HyPoradise/Chen et al., 2024; Whispering LLaMA/Radhakrishnan et al., 2023). The paper applies LoRA fine-tuning of an LLM to medication names — a straightforward application of existing techniques to a new domain. No novel architectural modifications, training objectives, or domain adaptation strategies are introduced. Prior GER work already demonstrated this paradigm; the paper does not explain what specifically makes their approach better suited for medication names.

- **Dataset poorly described and extremely small**: The "Pharma-Speak" dataset of ~600 medication names (506 train, ~94 test) is described only as "an open source dataset which had about 600 medication names prescribed globally with their trade names which we curated ourselves." There is no description of audio data, speaker characteristics, accent diversity, noise conditions, or whether this is even a speech dataset at all — it appears to be a text-only medication name list. The paper provides no information on how N-best hypotheses are generated from this data for training the LLM, leaving the experimental pipeline ambiguous.

### Minor

- **Model name inconsistency**: The abstract and Figure 1 reference "LLaMA 3" but Section 4.1 specifies "Llama-2-8b Instruct." This creates confusion about which model was actually used.
- **Missing experimental details**: No description of the prompt format for the LLM, the objective function (language modeling loss vs. classification over N-best), or how N-best hypotheses are formatted as input. No analysis of oracle N-best coverage (how often the correct transcription appears in the 10-best list).
- **No error analysis or qualitative examples**: The paper provides no examples of corrected transcriptions, no confusion analysis, and no categorization of error types (e.g., near-homophone confusions, abbreviation errors).

### Trivial

- The paper states "Code and dataset are available upon request" rather than being publicly available, which limits reproducibility but is a minor concern relative to the other issues.

## Nice-to-Haves

- Report WER (both overall and for medication entities specifically) alongside ROUGE, so results can be compared to standard ASR baselines and prior GER work.
- Include comparisons with simpler rescoring methods (e.g., n-gram LM rescoring, shallow fusion) to establish whether LLM-based rescoring provides meaningful benefits over cheaper alternatives.
- Analyze oracle WER (WER of the best hypothesis in the N-best list) to show the upper bound achievable by rescoring.
- Evaluate on existing clinical ASR benchmarks (e.g., AfriSpeech, MEDIC) for comparability with prior work.

## Removed Points

- **"Code not publicly available" as a fatal reproducibility concern**: While not ideal, this is standard in some communities and is minor relative to the fundamental evaluation problems. Kept as a trivial concern only.
- **"Overfitting concern due to 506 training samples on 8B model"**: While reasonable to flag, the paper does use LoRA with rank 4, which severely restricts trainable parameters. The concern is valid but secondary to the more fundamental metric and evaluation issues.
- **"Need for confidence intervals/statistical significance"**: With only ~94 test examples, this would be ideal but is standard practice in this domain to report at least variance, which is a subset of the broader evaluation inadequacy concern already captured above.

## Novel Insights

The paper's core observation — that LLM-based rescoring could correct medication-specific ASR errors in low-resource settings — has merit as a research direction. However, the execution and evaluation are too incomplete to substantiate this direction. The most interesting design insight (preserving LLM general knowledge via LoRA while domain-adapting for medication correction) is mentioned but not empirically validated through ablation.

## Suggestions

1. **Replace ROUGE with WER and report both side-by-side**: Evaluate the full pipeline (Whisper → LLM rescoring) with WER to enable direct comparison with standard ASR baselines.
2. **Add proper baselines**: Include Whisper-only WER, Whisper with simple n-gram rescoring, and comparison with prior GER methods (HyPoradise, Whispering LLaMA) on the same data.
3. **Report full training curves with error bars**: Show all 15 epochs, include variance across seeds, and explain the non-monotonic behavior observed.
4. **Describe the complete pipeline**: Clearly specify how training data is constructed from medication names (are these text-to-text pairs? is speech actually involved?), the prompt format, and the N-best decoding process.

## Score and Decision

**Calibration comparison:**
- **CPLLM** (scores 1, 1, 3, 3 — Reject): Similar pattern of applying LLM+LoRA to a medical task with weak baselines and marginal novelty. This paper has even weaker evaluation.
- **SLM Exploration** (scores 3, 3, 3, 3 — Reject): Similar lack of novelty, applying existing methods without meaningful adaptation.
- **Luganda MT** (scores 3, 5, 3, 3 — Reject): Similar low-resource domain adaptation with synthetic/small dataset and methodological concerns.
- **UADF/RobustGER** (scores 8-10 — Accept): These are proper GER papers with novel methods, thorough evaluation including WER, and comprehensive baselines — far above this paper's contribution.

The paper's fundamental evaluation problems (claiming WER improvement but reporting only ROUGE on a tiny dataset with no baselines), combined with minimal novelty, place it well below the acceptance threshold. It shares the core weaknesses of papers scored 2-3 (reject-grade) in similar domains.

MY FINAL SCORE: <pineapple>2</pineapple>
MY FINAL DECISION: <orange>Reject</orange>