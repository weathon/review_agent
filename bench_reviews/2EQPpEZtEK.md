## Summary
DISTAR introduces a zero-shot text-to-speech framework that couples an autoregressive language model with a masked diffusion model, operating entirely in a discrete residual vector quantization (RVQ) token space. The hybrid design aims to achieve blockwise parallelism, mitigate autoregressive exposure bias, and provide explicit controllability through features like RVQ layer pruning. The paper demonstrates state-of-the-art or competitive results on standard zero-shot TTS benchmarks.

## Strengths
- **Strong empirical performance:** DISTAR achieves leading or competitive scores on LibriSpeech-PC and Seed-TTS in key metrics (WER, SIM, UTMOS) and subjective evaluations (CMOS, SMOS), validating its effectiveness.
- **Practical and well-motivated design:** The method includes several impactful features: stochastic RVQ layer truncation during training enables test-time bitrate and compute control without retraining; the fully discrete pipeline eliminates the need for a separate duration predictor or forced alignment.
- **Comprehensive ablation studies:** The paper provides ablations on patch size, classifier-free guidance strategies, and decoding methods, offering clear justification for design choices.

## Weaknesses
- **Missing critical ablation to validate the hybrid approach:** The paper lacks a direct comparison to a strong, pure-autoregressive (AR) baseline trained on the same RVQ tokens and dataset. This is essential to substantiate the core claim that coupling AR with masked diffusion provides benefits over pure AR modeling for RVQ-based TTS.
- **Insufficient evidence for computational efficiency claims:** While the paper states DISTAR maintains "inference cost close to its continuous counterpart," it provides no direct comparison of inference latency, throughput, or FLOPs against key baselines (e.g., DiTAR, F5TTS). The main quality comparison uses DISTAR with NFE=24 versus DiTAR with NFE=10, making the efficiency claim unsupported.
- **Technical ambiguity in the training formulation:** The use of overlapping context windows (stride S < patch size P) is not fully reconciled with the likelihood factorization in Equation (1). It is unclear how tokens within overlapping conditioning contexts are handled during training, which could affect reproducibility and the theoretical clarity of the training objective.
- **Reliance on heuristic decoding strategies:** The proposed layer-wise and position-wise temperature shaping, along with a hybrid sampling schedule, are presented as empirical fixes for a "tail-first bias." While they improve results, their necessity suggests a potential modeling bias or optimization issue, and they are not derived from or justified by the underlying masked diffusion framework.
- **Statistical significance of subjective improvements is unclear:** The reported CMOS improvement (0.22 ± 0.13 for DISTAR vs. 0.01 ± 0.12 for F5TTS) has overlapping confidence intervals. Without statistical significance testing, the claimed superiority in naturalness is not fully substantiated.

## Nice-to-Haves
- More thorough analysis of hyperparameter sensitivity (e.g., stride S, number of diffusion steps) beyond the patch size ablation.
- Evaluation on challenging, out-of-distribution or very long-form prompts to better validate the claimed robustness.
- Deeper qualitative or quantitative analysis of the division of labor between the AR drafter and the diffusion refiner (e.g., error rates by token position/layer).
- Clarification in Figure 2 that the x-axis "RVQ Layers" refers to the total number of layers used during inference (i.e., 9 - ℓ pruned layers).

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Criticism that the iterative masked diffusion decoding is "heuristic" and lacks theoretical linkage:** The method follows established masked diffusion frameworks (e.g., LLaDA, D3PM) and the described confidence-based decoding is a standard practical approach for such models.
- **Criticism about data dependency and reproducibility due to use of the proprietary Emilia dataset:** While a limitation, the use of large-scale, curated datasets is common in contemporary TTS research and does not constitute a methodological flaw specific to DISTAR.
- **Criticism demanding a user study or theoretical proofs:** These are not standard requirements for an algorithmic/engineering contribution in this domain.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add an experiment comparing DISTAR to a strong pure-autoregressive RVQ baseline (e.g., a VALL-E 2 style LM trained on the same tokens) to isolate the benefit of the hybrid AR-diffusion design.
- Include standardized inference latency/throughput benchmarks (e.g., real-time factor, tokens/sec) comparing DISTAR to key baselines under controlled hardware settings.
- Clarify the training objective and how overlapping context windows are handled in the likelihood computation (Section 3.1.1).
- Perform statistical significance testing (e.g., t-test) on the subjective evaluation scores to confirm the reported improvements.