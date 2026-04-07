=== CALIBRATION EXAMPLE 75 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the core contribution of a noise-robust semantic speech tokenizer. The abstract clearly states the problem (fragility of current tokenizers), the proposed solution (StableToken with multi-branch architecture and consensus training), and the key results (significant UED reduction and downstream improvements). The claims are specific and appear to be supported by the later experimental sections.

### Introduction & Motivation
The problem is well-motivated, citing the rise of SpeechLLMs and the critical role of semantic tokenizers. The observation that existing tokenizers are fragile to "meaning-irrelevant acoustic perturbations" is compelling. The introduction effectively critiques the "architectural flaw" (single-path quantization) and "distant supervisory signal" (ASR loss indifference) as root causes. The contributions of a co-designed architecture (voting-LFQ) and training strategy (noise-aware consensus) are clearly stated. The failure analysis of naive solutions (offline ensemble, token-level consistency loss) strengthens the motivation.

### Methods / Approach
**Overall Structure & Voting-LFQ Module:** The method is clearly described and appears reproducible. The Voting-LFQ module is a non-trivial extension of LFQ. The use of an odd number of branches for bit-wise majority voting is clever and theoretically justified. The explanation of how bit-level voting can correct token-level errors is a key insight. The complexity analysis in Appendix B.6 is thorough and convincingly shows negligible overhead.

**Noise-Aware Consensus Training:** The strategy of feeding clean audio to a majority of branches and perturbed audio to a minority is innovative. The consensus loss (Eq. 4) applied to pre-quantization continuous vectors is a well-motivated choice to avoid discrete optimization issues. The justification for using L2 loss over cosine similarity in Appendix B.4 is reasonable.

**Potential Gaps/Questions:**
1.  **Training Stability:** The method relies on a multi-component loss (Eq. 5). While results are strong, more discussion on the sensitivity to the hyperparameters λ1, λ2, λ3 and the stability of the joint optimization would be beneficial.
2.  **Branch Diversity:** The consensus loss pulls branches together. Is there a risk of collapse where all branches learn identical projections, negating the ensemble benefit? The paper argues the noisy inputs prevent this, but a more explicit analysis (e.g., measuring similarity between branch projections `p_i`) would strengthen the claim.
3.  **Perturbation Strategy:** Appendix B.3 details the noise types. However, the random selection of *a minority subset of k branches* per sample is mentioned but not fully elaborated. Is `k` fixed or sampled from a range? Could a dynamic strategy (e.g., based on input difficulty) be more effective?
4.  **Vocabulary Size & Quantization:** The model uses a vocabulary of 8192 (d=13). The paper correctly notes this makes robustness harder. However, the relationship between codebook size, bit depth `d`, and the efficacy of bit-wise voting could be explored further. Does the method scale favorably to even larger codebooks?

### Experiments & Results
**Tokenizer-Level Performance (Sec 4.1):** The evaluation is comprehensive. Table 1 shows dramatic UED improvements across all noise types, including OOD noise, which is convincing evidence of generalization. Table 2/G.24 shows that this robustness does not come at the cost of reconstruction quality; StableToken is competitive or superior in WER/MOS. This addresses a critical counter-argument.

**Downstream Task Performance (Sec 4.2):** The controlled, isogenic setup (same LLM backbone, fine-tuning paradigm) ensures a fair comparison. The results are compelling:
*   **ASR (Fig 3, Table 3):** The widening performance gap as SNR decreases is a powerful demonstration of downstream robustness. Strong results on CHiME-4 reinforce real-world applicability.
*   **SER (Fig 3):** Consistent gains across noise levels show the tokenizer preserves paralinguistic cues robustly.
*   **TTS (Table 3):** Improved WER and MOS confirm the tokens are high-quality for generation.

**Ablation & Analysis (Sec 4.3 & Appendices):**
*   Table 4 provides a clean sequential ablation validating each component.
*   Table 5 and Appendix B.6 justify the choice of N=5 effectively.
*   Table 6 is an excellent case study that visually demonstrates the bit-wise error correction mechanism.
*   Appendix C provides valuable insight into the trade-off between quantization depth (semantics vs. acoustics) and the choice of noise ratio.
*   Appendix B.7 (cross-lingual analysis) is a strong addition, showing the model learns language-specific token distributions efficiently.

**Potential Concerns:**
1.  **Baseline Comprehensiveness:** The baselines cover major categories (SSL, distilled, supervised). However, a direct comparison to the most relevant prior work on robust tokenization—**NAST** and **R-SPIN**—is somewhat buried in the large Table 1. A focused discussion on why StableToken's supervised, voting-based approach outperforms these specifically designed robust tokenizers would be valuable.
2.  **Statistical Significance:** The improvements are large, but reporting confidence intervals or standard deviations for key metrics (e.g., WER on CHiME-4) would bolster claims.
3.  **Real-World Noise Diversity:** While ESC-50/10 and CHiME-4 are used, an evaluation on a completely unseen, challenging real-noise dataset (e.g., from a dashboard camera or crowded café) would further test generalization.
4.  **Long-Form Audio:** Appendix B.8 addresses chunking and boundary stability. The analysis shows consistent UED across segments, which is reassuring. However, the potential for error propagation or drift in very long conversations (e.g., multi-turn dialogue) is not explored and could be noted as a limitation.

### Writing & Clarity
The paper is generally well-written. The narrative is logical, and figures support the text. Some minor issues from the parser exist (e.g., garbled numbers in Figure 3, misplaced table lines), but these do not impede understanding. The description of the consensus loss and the bit-wise voting mechanism is particularly clear.

### Limitations & Broader Impact
A **significant weakness** is the lack of a dedicated "Limitations" section. The paper thoroughly analyzes its components but does not formally discuss broader constraints. The following limitations should be explicitly acknowledged:
1.  **Assumption of "Meaning-Irrelevant" Noise:** The method aims for invariance to acoustically perturbing but semantically irrelevant noise. It does not address adversarial perturbations designed to change meaning, nor does it handle noise that *does* change semantic content (e.g., overlapping speech). The scope should be clarified.
2.  **Training Complexity:** The multi-branch architecture and multi-view training, while efficient at inference, increase training cost and complexity compared to a single-path model.
3.  **Dependency on Clean Data:** The noise-aware consensus training relies on access to clean audio for the majority of branches. Performance in environments where only noisy audio is available for training (e.g., in-the-wild data) is not verified.
4.  **Broader Impact** is not discussed. While the work is largely technical, a brief statement on potential positive (more reliable speech assistants) and negative (potential for misuse in generating robust deepfake audio) impacts would be appropriate for ICLR.

### Overall Assessment
StableToken presents a novel, well-motivated, and effectively executed solution to a clearly identified problem in semantic speech tokenization. The core ideas—bit-wise voting LFQ and noise-aware consensus training—are innovative and synergistic. The experimental validation is extensive, covering intrinsic tokenizer properties and multiple downstream tasks with convincing gains, especially under noise. The main weaknesses are the absence of a formal limitations section and some missing analysis on training dynamics and branch diversity. These do not undermine the core contribution but should be addressed. The paper makes a strong case for the importance of tokenizer robustness as a foundation for resilient SpeechLLMs and proposes a method that significantly advances the state-of-the-art. It meets the high standards of ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces StableToken, a novel semantic speech tokenizer designed for enhanced robustness to acoustic noise. The core innovation is a multi-branch quantization architecture (Voting-LFQ) combined with a Noise-Aware Consensus Training strategy, which uses bit-wise majority voting and a consensus loss to produce stable token sequences even under perturbations. The method significantly reduces token instability (measured by Unit Edit Distance) and demonstrates improved downstream performance for SpeechLLMs on tasks like ASR, SER, and TTS under noisy conditions.

### Strengths
1. **Clear Problem Identification and Motivation**: The paper convincingly identifies a critical, underexplored flaw in contemporary semantic tokenizers: their surprising fragility to meaning-irrelevant acoustic noise, even at high SNRs. This is well-motivated with references to prior work (e.g., Messica & Adi, 2024) and clearly linked to downstream SpeechLLM degradation.
2. **Novel and Technically Sound Solution**: The proposed Voting-LFQ module and consensus training strategy are novel co-designs. The bit-wise voting mechanism is a clever solution that provides finer-grained error correction than token-level voting, and the training strategy (feeding perturbed inputs to a minority of branches) elegantly provides the necessary supervisory signal for invariance. The ablation studies (Table 4, 5) robustly validate each component's contribution.
3. **Extensive and Convincing Evaluation**: The paper provides comprehensive benchmarks at the tokenizer level (noise robustness, reconstruction fidelity) and downstream task level (ASR, SER, TTS across multiple noise conditions and datasets). The results are state-of-the-art, showing dramatic UED reductions (e.g., from 26.17% to 10.17% vs. S³ Tokenizer) and consistent downstream gains, especially as noise severity increases (Figure 3). The inclusion of OOD noise tests strengthens the claim of generalizability.
4. **Strong Reproducibility and Clarity**: The paper is well-structured, the method is clearly explained, and the appendices provide extensive details on datasets, hyperparameters, noise profiles, and computational efficiency. The commitment to release code and models further supports reproducibility, aligning well with ICLR expectations.

### Weaknesses
1. **Limited Discussion of Voting Mechanism Limitations**: While the bit-wise voting is powerful, the paper does not deeply analyze its failure modes. For instance, what happens under adversarial noise patterns designed to systematically flip specific bits across a majority of branches? A discussion on the theoretical or empirical limits of this "sparse bit-flip" assumption would strengthen the analysis.
2. **Real-World Applicability and Efficiency Trade-offs**: Although computational overhead is analyzed (Appendix B.6) and shown to be minimal, the evaluation focuses on controlled synthetic and real noise datasets. There is no demonstration on truly "in-the-wild" noisy audio (e.g., from challenging acoustic environments with reverberation, overlapping speech). Furthermore, while memory footprint is lower, the energy efficiency of parallel branch computation during inference is not discussed, which can be a practical concern.
3. **Superficial Comparison with Some Baselines**: The comparison with "semantic distilled tokenizers" (e.g., SpeechTokenizer, X-Codec) in Table 1 shows they are less robust, but this is somewhat expected as they are optimized for different trade-offs (acoustic fidelity). The paper could better contextualize that StableToken's primary advantage is within the "supervised semantic tokenizer" paradigm most relevant for SpeechLLMs, rather than claiming blanket superiority over all tokenizer types.
4. **Potential Overfitting to Noise Augmentation Strategy**: The training uses a specific set of stochastic perturbations (Gaussian, Pink, Brown noise, Bit Crush, Real-world noise from specific datasets). While results on OOD noise (ESC-10) are good, the model's robustness to entirely unseen perturbation types (e.g., codec distortions, fast fading, non-stationary interference) remains an open question. A broader noise robustness analysis would be beneficial.

### Novelty & Significance
**Novelty**: The work introduces a novel architectural paradigm (multi-branch voting-LFQ) and a corresponding training strategy (noise-aware consensus loss) specifically tailored for tokenizer stability. While ensemble methods and consistency training are known concepts, their application to solve the discrete token instability problem in speech tokenization via bit-level voting is novel and non-trivial.

**Significance**: The significance is high. Tokenizer instability is a fundamental bottleneck for deploying robust SpeechLLMs in real-world scenarios. By directly addressing this at the tokenizer level, the work provides a foundational improvement that cascades to multiple downstream tasks. The demonstrated performance gains under noise are substantial. This work could set a new standard for building noise-resilient speech processing pipelines.

### Suggestions for Improvement
1. **Analyze Failure Cases and Robustness Boundaries**: Include a dedicated analysis or discussion section on scenarios where the voting mechanism might fail (e.g., correlated bit errors across branches, extreme noise conditions). This would provide a more complete understanding of the method's limitations.
2. **Expand Real-World Evaluation**: Supplement the benchmark results with a small-scale qualitative evaluation on truly uncontrolled, real-world noisy audio clips (e.g., from YouTube, public speeches). This would strengthen the claim of practical utility.
3. **Deeper Analysis of Cross-Lingual and Prosodic Robustness**: Appendix B.7 analyzes token distribution across languages but does not explicitly evaluate noise robustness for low-resource or tonal languages. Furthermore, while SER results imply prosody preservation, a direct evaluation of prosodic stability (e.g., pitch, duration consistency under noise) would be valuable.
4. **Discuss Integration and Scalability**: Briefly discuss the ease/difficulty of integrating StableToken into existing SpeechLLM frameworks beyond the tested Qwen2.5-3B setup. Additionally, comment on the scalability of the multi-branch design to even larger models or vocabularies.
5. **Clarify Figure 1**: The caption and description of Figure 1 are slightly confusing. The bars in the figure seem to represent UED values from Table 1, but the labels (e.g., "38.66", "26.17", "10.17") could be more explicitly connected to the corresponding models in the text for immediate clarity.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the consensus loss formulation and alternatives.** The paper uses an L2 loss for consensus but does not justify why this is better than, e.g., contrastive or cosine similarity losses, which are common for invariance learning. Without this, it's unclear if the choice is optimal or a trivial contributor to success.
2. **Comparison to simple data augmentation and adversarial training baselines.** The paper claims architectural novelty is key, but does not show that simply training a single-branch tokenizer with heavy noise augmentation (or adversarial training on tokens) cannot achieve similar robustness. This is a critical baseline to establish the necessity of the multi-branch voting design.
3. **Evaluation on code-switching or accented speech.** The tokenizer is evaluated on clean and noisy but standard speech. Its performance on semantically challenging but acoustically varied inputs (e.g., code-switching, strong accents) is missing. This is important for claims of "semantic" robustness.
4. **Direct ablation of the bit-wise voting vs. token-level voting.** The case study suggests bit-wise is superior, but there is no quantitative ablation comparing the two voting schemes. This is necessary to validate the core technical claim about bit-level error correction.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of what information is lost due to increased stability.** The paper shows improved UED and downstream task performance, but does not analyze if the stability comes at the cost of losing useful para-linguistic information (e.g., subtle emotional cues, speaker identity). An analysis of speaker verification or emotion recognition using the tokens directly would reveal this trade-off.
2. **Failure mode analysis: when does voting fail?** The paper shows a success case. A systematic analysis of examples where UED remains high even with StableToken is needed. This would reveal the limitations of the method and whether failures correlate with specific noise types or phonetic contexts.
3. **Interpretability of the learned codebook.** The paper mentions cross-lingual efficiency in an appendix, but no analysis of whether the codebook clusters correspond to linguistically meaningful units (e.g., phonemes) and how noise affects those clusters. This is key for a "semantic" tokenizer claim.

### Visualizations & Case Studies
1. **Visualization of token sequence alignments (like Figure 1) for challenging real-world noisy samples.** Figure 1 is a schematic. Actual token sequence alignments (e.g., a matrix showing token matches/mismatches over time) for real noisy audio would convincingly show the stability compared to baselines.
2. **Case studies on downstream task failures.** Show examples where baseline tokenizer fails on a noisy ASR or TTS task and StableToken succeeds, alongside the corresponding token sequences. This would directly link token stability to downstream performance.
3. **Visualization of the bit-level error patterns across branches.** A heatmap showing which bit positions are most frequently flipped by noise across branches would provide insight into the voting mechanism's effectiveness and potential vulnerabilities.

### Obvious Next Steps
1. **Investigate the interplay between the number of branches (N) and the complexity of noise.** The ablation picks N=5, but does not study if optimal N changes with noise type/SNR. A analysis of robustness vs. N across noise profiles would be more insightful.
2. **Apply StableToken to a real streaming SpeechLLM scenario.** The paper uses chunking but does not evaluate in a true streaming setting with overlapping windows, where boundary stability is critical. This is a natural next step for a robustness claim.
3. **Combine with acoustic tokenizers for full robustness.** The work focuses on semantic tokens. A clear next step is to integrate this with a robust acoustic tokenizer (like NAST) for a complete noise-robust speech representation, which should be discussed.

# Final Consolidated Review
## Summary
StableToken introduces a novel semantic speech tokenizer designed for noise robustness. It combines a multi-branch quantization architecture (Voting-LFQ) that performs bit-wise majority voting with a Noise-Aware Consensus Training strategy. This co-design achieves state-of-the-art token stability, drastically reducing token sequence changes under noise, which directly translates to improved robustness of downstream SpeechLLMs in ASR, SER, and TTS tasks.

## Strengths
- **Novel and well-motivated co-design of architecture and training.** The Voting-LFQ module provides intrinsic fault tolerance through bit-wise majority voting, which can correct errors even when a majority of branches are corrupted at the token level. This is synergistically paired with a consensus training loss that explicitly enforces invariance by feeding perturbed audio to a minority of branches, using the clean majority as a stable anchor. The ablation studies (Tables 4, 5) robustly validate the necessity of each component.
- **Extensive and convincing experimental validation.** The paper demonstrates a dramatic improvement in intrinsic tokenizer robustness (e.g., average UED reduced from 26.17% to 10.17% vs. a top supervised baseline) while maintaining or improving reconstruction fidelity (Table 2). This foundational stability leads to significant downstream gains, especially under severe noise, as shown in controlled, isogenic evaluations across ASR (CHiME-4, noise-augmented LibriSpeech), SER, and TTS tasks (Figure 3, Table 3). The inclusion of out-of-distribution noise tests and cross-lingual analysis (Appendix B.7) strengthens claims of generalization.
- **Practical efficiency and strong reproducibility.** Detailed analysis (Appendix B.6) shows the multi-branch design adds negligible parameters and computational overhead during inference, with competitive or better latency and memory footprint versus baselines. The paper provides comprehensive training details, hyperparameters, noise profiles, and commits to releasing code and models, ensuring reproducibility.

## Weaknesses
- **Insufficient analysis of the voting mechanism's failure modes and robustness boundaries.** The paper convincingly demonstrates successful error correction via bit-wise voting but does not systematically analyze when or why this mechanism might fail. A discussion or experiment exploring conditions that could break the "sparse bit-flip" assumption (e.g., adversarial noise patterns, correlated bit errors across branches, or extreme noise types not in the training distribution) would provide a more complete understanding of the method's limitations and scope.
- **Limited evaluation on truly in-the-wild, complex acoustic environments.** While evaluations on synthetic noise and curated real-noise datasets (ESC-50, CHiME-4) are strong, the paper lacks validation on completely uncontrolled, real-world audio featuring complex mixtures of noise, reverberation, and overlapping speech. This gap leaves the practical utility in the most challenging scenarios less certain.
- **Incomplete exploration of the trade-off between stability and para-linguistic information preservation.** The improved token stability and strong SER results suggest prosodic information is retained, but a direct, quantitative analysis is missing. An evaluation measuring how well the tokenizer preserves specific acoustic properties (e.g., pitch, speaking rate) under noise, or a controlled ablation on speaker identity retention, would clarify whether the drive for semantic stability inadvertently discards useful non-linguistic cues.

## Nice-to-Haves
- A more detailed comparison of the L2 consensus loss against alternative invariance losses (e.g., contrastive, cosine similarity) to further justify the design choice.
- Investigation into whether the optimal number of voters (N) is dependent on noise type or severity, rather than a fixed compromise.
- A brief discussion on the integration path for StableToken into diverse SpeechLLM architectures beyond the tested framework.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness: "Lack of a dedicated Limitations section."** While a formal section is absent, the paper discusses constraints in appendices (e.g., chunking for long audio in B.8, trade-offs of quantizer depth in C). The core limitations are better captured as specific weaknesses above.
- **Weakness: "Training stability and sensitivity to hyperparameters."** The paper provides full hyperparameters (Table 8) and shows stable, superior results across many experiments. No evidence is presented that training is unusually sensitive.
- **Weakness: "Risk of branch collapse due to consensus loss."** The training strategy explicitly prevents this by feeding different (clean vs. noisy) inputs to different branches, creating a necessary diversity that the consensus loss must bridge.
- **Weakness: "Need for comparison to adversarial training or heavy augmentation baselines."** The paper's core claim is an architectural innovation. Demanding that it also outperform all possible enhanced training baselines for a single-path model is scope creep. The sequential ablation (Table 4) already shows the multi-branch architecture is essential.
- **Weakness: "Demand for confidence intervals or statistical significance tests."** For the large-scale benchmarks reported (e.g., WER on CHiME-4, UED on FLEURS), single-run evaluation is standard in the field. The improvements are substantial and consistent across multiple noise types and tasks.
- **Weakness: "Requirement for evaluation on code-switching or accented speech."** The paper's scope is robustness to acoustic noise, not robustness to linguistic or dialectal variation. This is a different research question.
- **Strength: "The paper is well-written."** This is a generic strength that applies to many papers and is not specific to this contribution.

## Novel Insights
The key novel insight is the use of bit-level consensus within a multi-branch quantizer to achieve discrete token stability. Unlike token-level voting or ensembles, bit-wise majority voting can recover the correct token even when a majority of branches produce incorrect tokens at the token level, provided the underlying bit errors are sparse. This mechanism, coupled with a consensus loss applied to continuous pre-quantization representations, effectively decouples the learning of noise-invariant semantics from the brittle process of discrete quantization, offering a new paradigm for building robust discrete representations.

## Suggestions
- Add a dedicated subsection analyzing failure cases: examine samples where StableToken's UED remains relatively high, identify the noise conditions or phonetic contexts, and discuss whether the failure is due to violating the sparse bit-flip assumption or other factors.
- Conduct a small-scale but qualitative evaluation by applying StableToken to a few minutes of challenging, real-world noisy audio (e.g., from public videos in crowded settings) and compare token sequences and downstream task performance with baselines to bolster claims of real-world applicability.

# Actual Human Scores
Individual reviewer scores: [10.0, 6.0, 8.0, 6.0]
Average score: 7.5
Binary outcome: Accept
