=== CALIBRATION EXAMPLE 89 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title clearly reflects the paper's core contribution: a noise-robust semantic tokenizer for SpeechLLMs. The abstract concisely states the problem (fragility of existing tokenizers), the proposed solution (multi-branch consensus via bit-wise voting), and the key results (state-of-the-art token stability and downstream gains). Claims are specific and appear supported by the experiments. However, the abstract lacks any mention of limitations or potential drawbacks, which is acceptable but could be improved for balance.

### Introduction & Motivation
The introduction effectively motivates the problem by highlighting a critical yet overlooked vulnerability: semantic tokenizers are surprisingly fragile to minor acoustic noise, which disrupts downstream SpeechLLMs. The analysis of two root causes (brittle single-path quantization and distant ASR supervision) is insightful. The introduction also explains why naive solutions (e.g., offline ensembles, token-level consistency losses) fail, setting a clear need for the proposed approach. The contributions are implicitly stated but could be more explicitly enumerated.

### Methods / Approach
The method is clearly described in two synergistic components: the Voting-LFQ module and Noise-Aware Consensus Training.

**Voting-LFQ Module**: The multi-branch architecture with bit-wise majority voting is a clever design. The use of an odd number of branches and bit-level aggregation provides fine-grained error correction. The mapping to integer tokens is straightforward. Computational overhead is claimed negligible, supported by analysis in Appendix B.6.

**Noise-Aware Consensus Training**: The strategy of feeding clean audio to a majority of branches and perturbed audio to a minority, then enforcing consensus via an L2 loss on continuous projections, is well-motivated. It avoids the non-differentiability of discrete tokens. The augmentation details are in Appendix B.3.

**Potential concerns**:
1. **Choice of consensus loss**: The authors justify using L2 over cosine similarity in Appendix B.4, but this reasoning should be briefly summarized in the main text, as it is a non-obvious design choice.
2. **Selection of noisy branches**: The paper states that a random minority subset \(k\) (with \(k < n/2\)) receives perturbed input. However, it is unclear how \(k\) is chosen—fixed or sampled? This should be clarified.
3. **Dependence on clean reference**: The training requires a clean audio version to anchor the consensus. This limits applicability in fully unsupervised or noisy-only scenarios. The paper should acknowledge this limitation.
4. **Interaction with LFQ**: The method builds on LFQ (Lookup-Free Quantization). While the modifications are clear, it would be helpful to discuss whether the robustness gains are specific to LFQ or could generalize to other quantization schemes.

### Experimental Setup
The experimental design is comprehensive, covering tokenizer-level robustness (UED), reconstruction quality (WER, MOS), and downstream tasks (ASR, SER, TTS). The use of an isogenic downstream setup (same LLM backbone, fine-tuning protocol) ensures fair comparison.

**Potential concerns**:
1. **Training data transparency**: The tokenizer is trained on 150k hours of speech, mixing open-source and in-house data. The in-house data is not described, which hinders full reproducibility. While the open-source datasets are listed in Appendix B.1, the exact mix and preprocessing of in-house data remain unclear.
2. **Noise parameter choices**: The SNR levels and bit-depth ranges for synthetic perturbations (Table 21) are provided but not justified. A brief rationale for these ranges would strengthen the experimental design.
3. **Baseline comparability**: The baseline tokenizers vary in frame rates, codebook sizes, and training data. The paper acknowledges that comparisons are most meaningful within the same type, but the downstream results rely on fine-tuning each tokenizer with the same LLM. It is unclear if all tokenizers were fine-tuned with identical data and steps. Appendix F lists datasets but does not specify if the same subsets were used for each tokenizer.

### Results
**Tokenizer-Level Performance**: Table 1 shows dramatic improvements in UED across all noise types, including OOD real noise. StableToken reduces average UED from 26.17% (best supervised baseline) to 10.17%. Table 2 shows competitive reconstruction quality, with top-tier WER and MOS scores. However, the MOS results are mixed (e.g., lower than GLM-4-Voice on SEED-TTSEN but higher on SEED-TTSZH). The claim of "state-of-the-art reconstruction performance" is thus partially supported; a more nuanced statement is warranted.

**Downstream Performance**: Figure 3 and Table 3 demonstrate clear robustness gains in ASR and SER under increasing noise, and superior TTS performance. The consistent trend across tasks is convincing.

**Analysis**: The ablation study (Table 4) validates each component's contribution. The voter count analysis (Table 5) shows diminishing returns beyond \(n=5\). The case study (Table 6) effectively illustrates bit-wise error correction.

**Potential concerns**:
1. **Statistical significance**: The paper does not report variance or statistical significance tests, especially for downstream results where differences are sometimes modest. Multiple runs with standard deviations would strengthen the claims.
2. **MOS evaluation details**: The MOS methodology (number of listeners, rating protocol) is not described. This is important given the subjective nature of MOS.
3. **Long-form audio stability**: While Appendix B.8 analyzes boundary stability for chunked processing, the evaluation is limited to Gaussian noise. A more comprehensive analysis across noise types would be beneficial.

### Analysis (Section 4.3)
The analysis is thorough, with component ablation, voter count sensitivity, and a qualitative case study. However:
- The ablation does not compare bit-wise voting to token-level voting, which would highlight the advantage of the proposed granularity.
- The voter count analysis stops at \(n=7\); a curve showing performance vs. \(n\) (including larger \(n\)) would better illustrate diminishing returns.
- The analysis of quantizer depth and clean/noisy branch ratio (Appendix C) is informative but buried in the appendix. Key insights (e.g., trade-off between robustness and paralinguistic detail) should be summarized in the main text.

### Related Work
The related work adequately surveys semantic tokenizers and noise robustness literature. However, it could more critically position StableToken against prior robust tokenizers (e.g., NAST, R-SPIN), explicitly contrasting architectural and training differences that lead to superior performance.

### Conclusion
The conclusion summarizes the work but does not discuss limitations or future directions. This is a missed opportunity to provide a balanced perspective.

### Writing & Clarity
The paper is generally well-written and logically structured. Figures and tables are informative, despite some parsing artifacts (e.g., garbled text in Figure 1, strikethroughs in tables) that are clearly due to PDF extraction and not the authors' fault. A few sections are dense; some explanations (e.g., loss function choice) could be moved from the appendix to the main text for clarity.

### Limitations & Broader Impact
The paper lacks a dedicated limitations section. Key limitations include:
1. **Clean reference requirement**: Training relies on clean audio for the majority of branches, which may not be available in all scenarios.
2. **Large-scale training data**: The use of proprietary in-house data limits reproducibility.
3. **Task scope**: Downstream evaluation is limited to ASR, SER, and TTS; generalization to other SpeechLLM tasks (e.g., spoken question answering, dialogue) is untested.
4. **Computational cost**: While inference overhead is minimal, the pre-training cost (150k hours) is substantial.

Broader impact is not discussed. The authors should include a brief statement on potential societal implications, such as improved accessibility in noisy environments, but also risks like misuse for deepfakes or surveillance.

### Reproducibility Statement
The statement is included, and appendices provide extensive details on datasets, hyperparameters, and prompts. However, the in-house data is not described, and the code/model are promised but not yet released. This is acceptable if the release is guaranteed upon acceptance.

## Overall Assessment
StableToken presents a novel and compelling solution to the critical problem of tokenizer fragility in SpeechLLMs. The core idea—multi-branch quantization with bit-wise voting and consensus training—is innovative and well-executed. The experimental results are comprehensive and demonstrate state-of-the-art token stability and strong downstream robustness gains. However, the paper has notable weaknesses: lack of a limitations section, incomplete transparency regarding training data, and missing statistical significance analysis. Addressing these issues, particularly by acknowledging limitations and providing more reproducibility details, would significantly strengthen the paper. Given ICLR's emphasis on novel, impactful contributions with rigorous evaluation, the paper's strengths outweigh its weaknesses. With revisions, it would be a strong candidate for acceptance.

**Recommendation**: Accept, but encourage the authors to add a limitations section, clarify training data details, and include variance/statistical significance in the results.

# Neutral Reviewer
## Balanced Review

### Summary
This paper identifies a critical weakness in modern supervised semantic speech tokenizers: they are surprisingly fragile to acoustic noise, producing drastically different token sequences even under high-SNR perturbations where speech remains intelligible. This instability increases the learning burden for downstream SpeechLLMs. To address this, the authors propose StableToken, a tokenizer that introduces a multi-branch Voting-LFQ module (enabling bit-wise majority voting for robustness) coupled with a Noise-Aware Consensus Training strategy (using a clean-branch majority to stabilize noisy-branch representations). The method achieves state-of-the-art token stability and translates this robustness into significant downstream performance gains for SpeechLLMs on ASR, SER, and TTS tasks.

### Strengths
1. **Clear Problem Identification and Motivation**: The paper convincingly demonstrates the fragility of existing semantic tokenizers (high Unit Edit Distance under noise) and effectively argues that this is a fundamental bottleneck for real-world SpeechLLM robustness. The introduction and related work sections are well-written and establish a strong motivation.
2. **Novel and Well-Designed Method**: The co-design of the multi-branch Voting-LFQ architecture and the consensus-based training strategy is elegant and novel. The bit-wise voting mechanism is a clever solution that provides finer-grained error correction than token-level voting, and the training strategy of feeding clean inputs to a majority of branches creates a stable anchor for learning.
3. **Extensive and Convincing Experiments**: The evaluation is comprehensive, spanning (a) intrinsic tokenizer robustness (UED) under diverse synthetic and real-world noise, (b) reconstruction quality (WER, MOS), and (c) downstream SpeechLLM performance on ASR, SER, and TTS under varying noise levels. The consistent and often dramatic improvements (e.g., ~60% relative UED reduction) strongly validate the core claim. The inclusion of OOD noise tests and a detailed computational efficiency analysis (latency, memory) is commendable.
4. **Excellent Analysis and Ablation**: The paper includes a thorough ablation study (Table 4) confirming the contribution of each component, an analysis of the optimal voter count, and a insightful case study (Table 6) that qualitatively illustrates the bit-wise error correction mechanism. The appendices provide substantial additional detail on hyperparameters, datasets, and cross-lingual analysis.

### Weaknesses
1. **Theoretical Justification and Limiting Assumptions**: While empirically powerful, the paper provides limited theoretical insight into *why* bit-wise voting is superior to token-level voting or other ensemble methods. Furthermore, the consensus loss relies on an \(L_2\) distance between continuous projections; a more detailed discussion of this choice versus alternatives (e.g., contrastive losses) and its interaction with the STE-based binarization gradient flow would strengthen the methodological foundation.
2. **Incomplete Baseline Comparisons and Potential Confounds**: Some baseline tokenizers (e.g., GLM-4-Voice) operate at a lower frame rate (12.5Hz vs. 25Hz), which may inherently affect sequence stability metrics like UED. While the authors note that their larger vocabulary (8192 vs. 4096/6561) makes robustness harder, a more controlled comparison (e.g., matching frame rates or performing rate conversion) would further solidify the fairness of the comparison. The training data mix for StableToken, while detailed in the appendix, is not identically matched to all baselines, leaving a minor confound.
3. **Limited Exploration of Failure Modes and Long-Context Stability**: The analysis focuses on successes. A discussion of scenarios where StableToken might still fail (e.g., specific noise types, extremely low SNR) would provide a more complete picture. Additionally, while Appendix B.8 addresses chunk boundary stability, the evaluation is for 30-second chunks; the impact on very long-form audio processing (e.g., multi-minute context) and potential error propagation in an autoregressive SpeechLLM setting is not deeply explored.

### Novelty & Significance
**Novelty**: The work is highly novel. The integration of a multi-branch, bit-wise voting quantizer within a supervised semantic tokenizer framework and the associated noise-aware consensus training paradigm represent a significant architectural and algorithmic advance. While ensemble and voting ideas exist broadly, their application to solve the specific problem of discrete token instability in VQ-based speech tokenizers is new and clever.
**Significance**: The significance is substantial. Noise robustness is a critical requirement for deploying SpeechLLMs in real-world environments. By tackling the problem at the foundational tokenizer level, the paper offers a direct and effective path to more resilient systems. The demonstrated downstream gains across multiple tasks are compelling. The work is likely to influence future research on robust speech representation learning and has clear practical implications.

### Suggestions for Improvement
1. **Strengthen Theoretical Analysis**: Add a section or appendix providing a more formal analysis of the error-correcting capacity of bit-wise majority voting versus token-level voting, perhaps using concepts from coding theory or analyzing the probability of bit-flip recovery. Discuss the gradient properties of the consensus loss in relation to the non-differentiable quantization step.
2. **Conduct More Controlled Ablations on Frame Rate and Vocabulary Size**: To isolate the effect of the proposed method from hyperparameters, consider running an ablation where a baseline \(S^3\) Tokenizer is trained with a matched frame rate (25Hz) and a larger, matched vocabulary size (8192) to serve as a more direct comparison point.
3. **Explore and Discuss Failure Modes**: Include a qualitative analysis of a few failure cases—utterances where UED remains high despite StableToken or where downstream task performance degrades unexpectedly. This could reveal interesting boundaries of the method's robustness and suggest directions for future work.
4. **Clarify Long-Context/Streaming Implications**: For a conference like ICLR, where generative sequence modeling is central, briefly discuss the implications of the chunk-based processing strategy for streaming applications or very long audio. Could the consensus mechanism be applied across chunk boundaries in a streaming setup?

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1.  **Controlled ablation on the multi-branch ensemble effect.** The paper must isolate whether the gains stem from the novel bit-wise voting/consensus training or simply from ensembling multiple branches. A critical baseline is a multi-branch model with *independent* codebooks trained with standard LFQ and ASR loss (no consensus loss or shared voting), evaluated on the same data. Without this, the core architectural contribution is not convincingly demonstrated.
2.  **Equally scaled baseline on the same massive dataset.** The model is trained on 150K hours of data. A fundamental baseline is a single-branch (or standard *S³* Tokenizer) model trained from scratch on the **identical dataset and compute budget**. The impressive UED gains could be partly attributed to superior data scale rather than the proposed method.
3.  **Robustness to broader, real-world acoustic perturbations.** Evaluation is limited to additive noise and bit-crush. To claim general "noise robustness," test on common acoustic distortions like reverberation, convolutional noise, variable speed, and low-bitrate codec artifacts (e.g., OPUS, AMR). Their absence leaves the practical robustness claim incomplete.

### Deeper Analysis Needed (top 3-5 only)
1.  **Systematic analysis of bit-error sparsity and voting efficacy.** The claim that bit-wise voting recovers tokens even with majority branch errors is pivotal but supported only by anecdotal case studies. A quantitative analysis is needed: for erroneous tokens, plot the distribution of bit-level Hamming distances between branches and the consensus. This directly validates the core mechanism of exploiting sparse bit-flips.
2.  **Probing for potential loss of paralinguistic information.** The consensus loss could over-smooth representations, harming tasks needing fine-grained acoustic cues (e.g., SER, speaker ID). Perform a probing study: fit simple classifiers on frozen StableToken features for tasks like speaker, emotion, and pitch contour prediction. Comparing to baseline tokenizers reveals the trade-off between stability and acoustic detail preservation.
3.  **Analysis of failure modes.** When and why does StableToken fail? Provide a categorized analysis of high-UED examples (e.g., specific noise types, phonetic contexts). This is crucial for understanding the method's limitations and for future work.

### Visualizations & Case Studies
1.  **Spectrogram and token alignment visualizations for failure/success cases.** For key examples, show aligned spectrograms of clean/noisy audio alongside the token sequences from baselines and StableToken. Visual misalignments in baseline tokens versus stable alignments in StableToken would powerfully illustrate the problem and solution. Include cases where voting corrects errors and where it fails.
2.  **Bit-level agreement heatmaps across branches.** For a set of frames, visualize a heatmap of bit values (e.g., +1/-1) across all branches and the final voted output. This would intuitively show the consensus formation and how noisy branches deviate at the bit level, providing direct evidence for the proposed mechanism.

### Obvious Next Steps
1.  **Evaluate on a broader set of SpeechLLM backbones.** All downstream results use the same Qwen2.5-3B backbone. To claim general applicability, results with at least one other prevalent LLM (e.g., Llama, Gemma) are necessary to rule out architecture-specific synergies.
2.  **Benchmark inference speed and memory formally across platforms.** While an efficiency analysis is in the appendix, it should be moved to the main text and expanded. Compare latency, throughput, and memory against baselines not just on GPU but also on CPU and mobile-relevant hardware (e.g., ARM), as real-world deployment is a key motivation.
3.  **Explore adaptive mechanisms.** The paper uses a fixed 5 branches and a fixed clean:noisy ratio during training. An obvious step is to briefly explore or discuss the potential of adaptive voting (e.g., confidence-weighted voting) or dynamically adjusting the number of active branches based on estimated noise level.

# Final Consolidated Review
## Summary
This paper identifies a critical fragility in supervised semantic speech tokenizers: they produce unstable token sequences under minor acoustic noise, harming downstream SpeechLLMs. To address this, the authors propose StableToken, a tokenizer that combines a multi-branch quantization architecture with bit-wise majority voting and a noise-aware consensus training strategy. The method achieves state-of-the-art token stability and translates this robustness to significant gains in downstream speech understanding and generation tasks under noisy conditions.

## Strengths
- **Novel co-design of architecture and training:** The Voting-LFQ module with bit-wise majority voting provides inherent error correction, while the Noise-Aware Consensus Training explicitly enforces stability using a clean-audio anchor. This synergistic approach is elegantly motivated and effectively addresses the dual causes of tokenizer fragility.
- **Comprehensive and convincing evaluation:** The paper demonstrates superior noise robustness (e.g., ~60% relative reduction in Unit Edit Distance), maintains competitive reconstruction quality, and shows consistent downstream improvements in ASR, SER, and TTS under diverse noise conditions. The isogenic downstream setup ensures fair comparisons.
- **Thorough analysis and ablation:** Component ablation validates each design choice, voter count analysis justifies practical configuration, and a qualitative case study clearly illustrates the bit-wise error correction mechanism. Appendices provide extensive details on datasets, hyperparameters, and efficiency.

## Weaknesses
- **Insufficient isolation of the core mechanism:** The paper does not include a controlled ablation comparing StableToken to a multi-branch model with independent codebooks trained only with the ASR loss (no consensus loss or bit-wise voting). Without this, the contribution of the novel consensus mechanism versus simple ensemble effects is not fully substantiated.
- **Limited reproducibility due to undisclosed training data:** While open-source datasets are listed, the tokenizer is trained on 150k hours of speech including proprietary in-house data that is not described. This hinders full reproducibility and fair comparison of data scaling effects.
- **Absence of a limitations section:** The paper omits discussion of key constraints, such as the training dependency on clean audio for the consensus anchor, potential edge cases in failure modes, and the impact of fixed chunking for long-form audio beyond the analyzed boundaries.

## Nice-to-Haves
- Quantitative analysis of bit-error sparsity across branches to formally validate the voting mechanism's error-correction capacity beyond anecdotal examples.
- Probing studies to explicitly verify that paralinguistic information (e.g., speaker identity, fine-grained prosody) is preserved despite the consensus-driven smoothing.
- Exploration of robustness to a broader set of real-world acoustic perturbations like reverberation or low-bitrate codec artifacts.

## Novel Insights
The key novel insight is that tokenizer instability under noise can be effectively addressed through a consensus-driven paradigm operating at the bit level. By integrating a multi-branch quantizer with bit-wise voting and a training strategy that uses clean branches to stabilize noisy ones, the paper demonstrates that discrete token sequences can be made remarkably invariant to acoustic perturbations without sacrificing semantic fidelity. This approach shifts the focus from post-hoc robustness techniques to foundational tokenizer design.

## Suggestions
- Add a limitations section addressing the clean-audio requirement for training, potential failure modes, and long-context processing implications.
- Include an ablation experiment with a multi-branch baseline without consensus loss to isolate the effect of the proposed voting and training strategy.
- Provide details on the MOS evaluation methodology (e.g., number of listeners, rating protocol) to improve transparency.

# Actual Human Scores
Individual reviewer scores: [10.0, 6.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
