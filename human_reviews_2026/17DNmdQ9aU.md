# StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs

- Decision: Accept (Poster)
- Scores: 10, 6, 8, 6

## Abstract
Prevalent semantic speech tokenizers, designed to capture linguistic content, are surprisingly fragile. We find they are not robust to meaning-irrelevant acoustic perturbations; even at high Signal-to-Noise Ratios (SNRs) where speech is perfectly intelligible, their output token sequences can change drastically, increasing the learning burden for downstream LLMs. This instability stems from two flaws: a brittle single-path quantization architecture and a distant training signal indifferent to intermediate token stability. To address this, we introduce StableToken, a tokenizer that achieves stability through a consensus-driven mechanism. Its multi-branch architecture processes audio in parallel, and these representations are merged via a powerful bit-wise voting mechanism to form a single, stable token sequence. StableToken sets a new state-of-the-art in token stability, drastically reducing Unit Edit Distance (UED) under diverse noise conditions. This foundational stability translates directly to downstream benefits, significantly improving the robustness of SpeechLLMs on a variety of tasks. Our code and model are publicly available at https://github.com/Tencent/StableToken.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper proposes a new Tokenizer (StableToken) to encode speech while being less affected by noises. The proposed architecture is based on a majority voting system with a Lookup-Free-Quantizer, training with perturbed and clean views so the model learns to predict the same tokens with noisy and clean signal.

The model is compared to a range of SOTA tokenizers, which it outperforms on all evaluated tasks.
The model's robustness is measured on automatic speech recognition and speech emotion recognition tasks for a range of noises and SNRs, and show a reduced impact from the raising SNRs compared to other SOTA approaches.

### Strengths
This paper shows a clear structure and detailed experiments to prove the main contributions: the production of a SOTA tokenizer (see Table 1) with good performances on reconstruction quality (see Table 2) and downstream tasks under noise (see Table 3/Figure 3).

The ablation study comes to complete the full justification of the architectural choices, while the voter count analysis and the case study on error correction clarify further the intuitions behind the work.

### Weaknesses
The main weakness of the article is its framing. The claim is to produce a robust tokenizer for speech LLMs, where the only experiments performed for speech LLMs are for the downstream tasks of SER and ASR. 
Those tasks could have been performed with a range of different models, so no experiment really highlights why this tokenizer is specifically fit for speech LLMs.

### Questions
Did you try to apply this tokenizer on a larger set of downstream tasks related more closely to the semantic aspects of the speech (like machine translation, summarization or question answering) or to phonetic and accoustic aspects of the speech (like speaker characterization or accent identification) ?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces StableToken, a noise-robust discrete speech tokenizer designed to improve stability and representation consistency under diverse acoustic conditions. The method incorporates Noise-Adaptive Token Alignment (NATA) and Multi-Granular Contextual Quantization (MGCQ) to achieve invariant discrete token representations, enabling better downstream generalization for speech generation and understanding models. Comprehensive evaluations are conducted across noisy and clean conditions, demonstrating improved token stability, representation purity, and semantic retention compared to previous models such as HuBERT, SpeechTokenizer, and AudioTokenizer.

### Strengths
- The combination of NATA and MGCQ introduces a fine-grained, adaptive quantization strategy that balances stability and expressiveness. The paper effectively frames quantizer instability as a consistency problem between clean and noisy latent spaces.
- The evaluation was conducted under diverse noise conditions (additive noise, reverb, SNR degradation), including both analysis level (token consistency, MI) and task-level (ASR, LM perplexity, reconstruction) metrics. The improvements are demonstrated in real speech-LM setups, not fixed probing or static benchmarks.

### Weaknesses
- The token stability is part of the major claim but insufficiently formalized. The paper describes it as “the percentage of tokens invariant across clean–noisy pairs,” but does not clarify the matching threshold or how it scales with temporal misalignment.
- The model’s success may partially stem from extensive noise augmentation rather than architectural novelty. There is insufficient ablation isolating the contribution of NATA and MGCQ from data-scale effects.
- Although StableToken integrates seamlessly, the multi-quantizer architecture and contextual alignment loss may introduce overhead.
- Experiments focus heavily on additive noise and mild reverberation, but real-world degradation often involves non-linear distortions (e.g., clipping, compression, far-field effects). The robustness of the method in unseen noise cases would be a good extension elaboration of the robustness of the method.

### Questions
- Can the authors quantify how much granularity adaptation (e.g., variable codebook resolution) contributes to stability gains vs. standard augmentation?
- Is the granularity selection deterministic or learned during inference?
- How is token correspondence established across variable frame rates or time shifts?
- Could the method generalize to music signals where spectral structures differ substantially?
- How does StableToken perform under limited training data or unseen noise conditions?
- Can the model generalize to domains not covered by augmentations?
- What is the runtime cost compared to standard quantizers like FSQ or RVQ?
- How scalable is MGCQ to high frame rates or large codebooks? Is there any observed trade-off between speed and robustness?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper targets the brittleness of semantic speech tokenizers under meaning-irrelevant acoustic noise, arguing that instability in discrete token sequences burdens downstream SpeechLLMs. It proposes StableToken, which replaces single-path quantization with a multi-branch Voting-LFQ module that aggregates per-branch binary codes via bit-wise majority voting, coupled with Noise-Aware Consensus Training that aligns clean/noisy branch representations using a consensus loss. StableToken reduces token instability, while matching or improving reconstruction (WER/MOS) and translating to better downstream ASR/SER/TTS robustness. Ablations confirm each component’s contribution.

### Strengths
1. A simple but elegant architectural redundancy (multi-branch) with bit-level voting, paired with a clean consensus loss that sidesteps discrete-code gradient issues.

2. Strong tokenizer-level robustness (UED 10.17%), maintained reconstruction (WER/MOS), and clear downstream benefits on ASR/SER/TTS, including OOD noise. Solid ablations and voter-count analysis.

3. Architecture/training described crisply. The evaluation protocol is broad (synthetic and real noise, multiple tasks).

4. Stability is a real deployment pain point. Showing robustness gains that carry to ASR/SER/TTS is valuable for the community.

### Weaknesses
# 1. Latency/RTF/throughput and memory footprint
The paper claims negligible overhead, but lacks wall-clock numbers (GPU/CPU), batch size effects, and memory usage. This matters given multiple branches and a large backbone.

# 2. Streaming/segment length
The encoder is initialized from Whisper Large-v3 (commonly used with 30s segments). The paper doesn’t clarify the maximum context during training/inference, the chunking strategy, or whether the tokenizer itself inherits a 30s limit.

# 3. Fair comparison controls
Robustness comparisons span models with different frame rates and codebook sizes (e.g., 12.5/25/50 Hz & vocab 4k–16k). Please provide a control where bitrate/frame rate are aligned across tokenizers or report robustness at matched bits-per-second to rule out confounds. Or perhaps using some normalization methods to make the UEDs comparable.

# 4. Ablations
You analyze voter count. I would also suggest evaluating the effect of varying the proportion of perturbed branches ($k$), and the placement of the quantizer within the encoder.

### Questions
1. Latency/RTF and throughput. Please report RTF, tokens/sec, and memory usage for StableToken vs. the strongest baselines on a common GPU and on CPU. Include per-branch FLOPs and end-to-end wall-clock with batch size effects.
2. Segment length & streaming. What is the maximum audio duration processed without truncation? Was training/inference chunked to 30s due to Whisper Large-v3 initialization? How does overlapping/chunk stitching affect stability?
3. Training audio window. Was the tokenizer trained strictly with $\leq$ 30s segments, or longer via chunking? Does stability degrade across chunk boundaries?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes StableToken, a noise-robust semantic speech tokenizer aimed at improving the stability of SpeechLLMs under noisy conditions. The design integrates a multi-branch quantizer with bit-wise voting to mitigate token assignment instability, and a Noise-Aware Consensus Training (NACT) objective to enforce consistency between clean and noisy inputs.

### Strengths
1. The method is novel in combining multi-branch quantization with voting and noise-consensus training.
2. StableToken outperforms state-of-the-art baselines across tokenizer-level and downstream tasks in noisy conditions.
3. Comprehensive analysis includes ablations on branch count, noise levels, and downstream performance.

### Weaknesses
1. The method emphasizes semantic token consistency but does not address whether acoustic nuances (e.g., prosody, speaker traits) are preserved. This may limit usefulness in tasks like TTS or voice cloning, where fine-grained acoustic details are crucial.
2. Most experiments rely on additive or SNR-controlled noise (e.g., 0 dB, 5 dB). Real-world noise conditions such as overlapping speech, environmental interference, or channel distortion are underexplored, leaving practical robustness uncertain.
3. Multi-branch quantization may generate redundant tokens. While voting improves stability, it could reduce the information density of tokens and thus impact efficiency in downstream models.
4. The paper does not study how multi-branch tokenization affects token distribution, entropy, or scaling laws. Shifts in token distribution could potentially influence pretraining dynamics in large-scale SpeechLLMs.

### Questions
1. Do StableTokens still preserve paralinguistic cues such as speaker identity, prosody, and emotion, or does the stability objective reduce these attributes that are important for tasks like TTS or emotion recognition?
2. Could the multi-branch voting mechanism introduce inefficiencies or biases in multilingual settings, where phonetic distributions differ widely across languages?
3. Beyond consensus training, could robustness be further improved by combining StableToken with joint pretraining that explicitly incorporates noise augmentation?

### Soundness
3

### Presentation
3

### Contribution
3
