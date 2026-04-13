=== CALIBRATION EXAMPLE 10 ===

# Final Consolidated Review
## Summary

TRACE proposes a structured output framework for video temporal grounding (VTG) tasks using video LLMs. Instead of generating unstructured natural language, TRACE represents outputs as sequences of events, each containing timestamps, salient scores, and captions, decoded autoregressively with separate encoders and heads for each modality. The approach is evaluated on dense video captioning, moment retrieval, and video highlight detection tasks.

## Strengths

- **Architecturally motivated solution to structured output generation:** The paper correctly identifies that standard LLM generation conflates timestamps, scores, and text in ways that don't match video structure. Using separate encoders and decoding heads (Sec 3.2.1) for timestamps, scores, and text—organized via task-interleaved sequence modeling—is a principled architectural response to this mismatch.

- **Strong empirical improvements on VTG tasks:** Table 2 shows substantial zero-shot gains over prior video LLMs (TimeChat, VTG-LLM): +6.5% Recall@1 IoU=0.5 on Charades-STA, +10.3% mAP on QVHighlights, and +3.1% CIDEr on Youcook2. The gains are consistent across three distinct VTG tasks.

- **Competitive fine-tuned performance:** After fine-tuning, TRACE achieves results competitive with task-specific methods on Youcook2 (Table 5: 35.5 CIDEr vs. Vid2Seq's 25.3) and approaches non-generative baselines on Charades-STA (61.7% vs. InternVideo2-6B's 70.0%), demonstrating that the architectural design transfers well to supervised settings.

## Weaknesses

- **Zero-shot evaluation validity unclear due to potential training data overlap:** Table 2 presents "zero-shot" results, but Section 3.3 states that Stage 2 training uses VTG-IT which the authors note includes data from QVHighlights and other VTG benchmarks. If the evaluation datasets (Youcook2, Charades-STA, QVHighlights) overlap with VTG-IT training data, the zero-shot characterization is inaccurate. The paper must explicitly confirm held-out status for each evaluation benchmark.

- **Ablation study has uncontrolled frame-count confound:** In Table 3, the "w/o causal event modeling" baseline uses 96 frames while "TRACE (VTG-IT)" uses 64 frames. Since performance increases monotonically with frame count (Table 3 lower block), the baseline receives an unfair advantage. The comparison should use identical frame counts to isolate architectural effects.

- **Complete failure of "w/o independent encoder/heads" baseline unexplained:** Table 3 reports "—" (complete breakdown) when using shared tokenizers instead of separate encoders/heads, with text stating it produces "irrelevant and meaningless responses." This result warrants investigation—was the baseline trained to convergence? Did training diverge? Vid2Seq and other works successfully add special time tokens to language model vocabularies; explaining why TRACE's approach fails catastrophically would strengthen the paper.

- **No evaluation of general video understanding capabilities:** TRACE specializes in VTG tasks through multi-head training. A natural concern is catastrophic forgetting of general video QA capabilities, yet no results on standard VideoQA benchmarks (MSVD-QA, ActivityNet-QA, VideoMME) are provided. For a paper claiming to advance "video LLMs" broadly, this omission is significant.

- **ActivityNet evaluation is in-distribution:** Table 4 evaluates on ActivityNet Captions but footnote indicates all models except Momentor were trained on this dataset. This should be clearly labeled as in-distribution evaluation, not compared against zero-shot Momentor* results as if fair.

- **Salient score contribution not validated:** The event formulation includes salient scores ($s_k$ in Eq. 1) as a core component, but no ablation removes this component to verify its necessity. If scores don't meaningfully contribute, the proposed "event" structure adds unnecessary complexity.

- **Theoretical claim in Footnote 1 unsupported:** The footnote states "Theoretically, the order of time, score, and text will not impact the results." This is incorrect for finite-capacity autoregressive models—different orderings affect training dynamics and learnability. No empirical ablation validates this claim.

## Nice-to-Haves

- **Efficiency analysis:** The multi-head architecture adds complexity; inference latency and memory overhead compared to standard video LLMs would help readers assess practical trade-offs.

- **Stopping criterion:** How does the model determine when to stop generating events? This is not explained in the main text.

- **Error propagation analysis:** In the autoregressive event framework, an incorrect timestamp prediction corrupts subsequent score and caption predictions. Understanding failure modes would strengthen the work.

## Removed Points

*These points are flagged to be removed, treat them with caution*

- **"Causal" terminology as misleading:** The term "causal" is standard in ML for autoregressive/left-to-right modeling (e.g., "causal attention," "causal language modeling"). This usage is conventional and not misleading.

- **Baseline vintage concern:** The baselines (TimeChat, VTimeLLM, VTG-LLM, Momentor, HawkEye) are from 2023–2024, which is appropriate for the paper's venue.

- **No statistical significance testing:** Point estimates without error bars are common practice in this venue and do not constitute a critical flaw.

## Novel Insights

The interleaved task-token sequence (Figure 3) effectively forces the model to maintain internal consistency between timestamps, saliency scores, and captions for each event—since subsequent tokens attend to all prior ones. This creates an implicit constraint that pure text generation cannot enforce: the model must internally align temporal grounding across modalities. The finding that shared tokenization causes complete breakdown (Table 3) suggests that the interference between numerical time tokens and semantic text tokens in a shared vocabulary is more severe than anticipated—potentially because digit tokens carry semantic meaning in natural language that conflicts with their use as numerical representations.

## Suggestions

1. **Clarify held-out status of evaluation datasets:** Explicitly state which evaluation benchmarks are excluded from VTG-IT and Stage 2 training data; if overlap exists, relabel results as "in-distribution" rather than "zero-shot."

2. **Fix ablation frame counts:** Rerun the "w/o causal event modeling" baseline with 64 frames for fair comparison, or explain the choice if intentional.

3. **Add VideoQA evaluation:** Include results on at least one standard video question-answering benchmark to verify that VTG specialization does not catastrophically degrade general video understanding.

4. **Investigate the shared-tokenizer failure:** Report training dynamics (loss curves, convergence status) for the "w/o independent encoder/heads" baseline to explain why it fails completely.

5. **Add salient score ablation:** Report results when removing the score prediction head to validate that this component contributes to performance.

# Actual Human Scores
Individual reviewer scores: [8.0, 5.0, 6.0, 8.0]
Average score: 6.8
Binary outcome: Accept
