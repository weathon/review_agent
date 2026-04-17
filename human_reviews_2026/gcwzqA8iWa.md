# Towards real-time BCI for speech with Whisper-based decoding of neural activity

- Decision: Reject
- Scores: 2, 4, 2

## Abstract
Decoding continuous speech from neural activity is a central challenge for brain–computer interfaces (BCIs), with major implications for restoring communication in individuals with paralysis. While recent work has achieved impressive performance using recurrent neural decoders trained on multi-electrode array (MEA) recordings, these models remain brittle, data-hungry, and struggle to generalize across sessions or participants. In this work, we introduce Whisper-BCI, the first neural speech decoder to integrate high-resolution MEA recordings with a large pretrained automatic speech recognition (ASR) model. Our approach leverages interpretability findings showing that Whisper’s encoder layers learn phoneme-selective representations with localized attention. Building on this insight, we adapt Whisper to predict phoneme embeddings from neural signals into the third layer of Whisper's encoder and fine-tune the model end-to-end with a hybrid objective combining CTC loss on phoneme alignments and cross-entropy loss on word tokens. We further introduce domain-informed modifications including windowed self-attention to capture articulatory continuity, day-specific low-rank projections to address non-stationarity and reduce computational complexity, and subject-specific input embedders for cross-subject training. Evaluated on Card et al. and Brain-to-Text '25 data, Whisper-BCI performs on par with or outperforms baselines relative to prior MEA decoders, and achieves cross-subject generalization, opening the door to robust decoding with limited resources. Post-processing with rescoring and grammar-guided correction yields an additional relative improvement, and the use of windowed attention has the potential to significantly reduce latency, enabling near-real-time online decoding. Our results demonstrate that pretrained ASR models can serve as effective language backbones for neural decoding and suggest a scalable path toward foundation models for speech BCIs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Whisper-BCI, a neural speech decoder that maps MEA recordings into a pretrained ASR model (Whisper) to improve robustness and cross-subject generalization. Evaluations are performed to show effectiveness.

### Strengths
* a) This paper is well organized and easy to follow.

### Weaknesses
**(a) Limited novelty**  
The proposed method builds directly on the  Whisper architecture. The proposed architecture still follows the CTC loss training with beam search for inference, with an additional CE loss for as the token-level loss being the only difference from Willett et al. As such, the work primarily represents an engineering integration of existing approaches rather than introducing new insights from either the neuroscience or the machine learning perspective.

**(b) Incremental improvement**  
The reported performance in Table 1 is only marginally better than the baseline models used in prior work, and these gains appear modest relative to the complexity added by the proposed approach. This raises questions about the practical advantage of the method over simpler existing techniques.

**(c) Insufficient Ablation Study**  
Ablation should at least be conducted to demonstrate the effectiveness of the proposed approach compared to the base model proposed by Willett et al. to show solidness.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose fine-tuning a Whisper model on intracranial MEA recordings to decode speech from the brain. They train their model to provide two decoding pathways: (1) where phoneme representations are rescored via WSFT, and (2) a more efficient pathway that leverages the Whisper decoder as an implicit language model and uses a small beam search to produce fast transcriptions. The authors run experiments where they jointly train with the Willett and Card data and show cross-subject generalisation and comparable performance to the current state-of-the-art on the respective datasets.

### Strengths
- Promising idea to combine both phoneme representations with word token predictions as they likely leverage complementary neural signals
- Aggregating all publicly available MEA datasets is a sensible step for scaling intracranial decoding
- Hierarchical normalization via month- and day-specific projections to account for representation drift is well-motivated
- Lightweight ASR approach by skipping the WSFT is sensible as the field moves towards real-time intracranial BCIs

### Weaknesses
- Improvements over Card and Willett are minor, if any, and are difficult to assess without error bars or a sense of statistical significance. Is this possible to gather via the competition platform?
- I would like to see a more robust analysis of subject generalisation as this could be an important aspect of this work. How does zero-shot subject transfer look? Does training on X% of Willett yield the same gain on Card as training on X% of Card?
- The authors have not made comparisons to the relevant baselines they discuss (e.g. LISA) and others described in the original Willett competition reflections paper [A]
- Minor: Line 105; NeuSpeech did not re-train Whisper on MEG recordings but rather fine-tuned the model with MEG.

I am open to raising my score if the authors can satisfactorily address the above weaknesses.

[A] Willett, F.R., Li, J., Le, T., Fan, C., Chen, M., Shlizerman, E., Chen, Y., Zheng, X., Okubo, T.S., Benster, T. and Lee, H.D., 2024. Brain-to-Text Benchmark'24: Lessons Learned. arXiv preprint arXiv:2412.17227.

### Questions
Questions
- Line 124-127: I understand how month- and day-specific projections help with representation drift across time, but how does it improve generalisation “across subjects”?
- Line 239-240: Although local window attention is a smart choice for computational efficiency, articulatory dynamics are not the only factor that may influence phoneme prediction. Low-frequency long-range semantic signals could inform and improve these predictions, too. Did the authors try training with full attention?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes Whisper-BCI, a brain–computer interface system that decodes speech directly from Brain-to-Text 2025 recordings using a modified version of the Whisper automatic speech recognition (ASR) model. The authors project neural features into Whisper’s encoder and optimize it with a hybrid loss that combines phoneme-level CTC loss and token-level cross-entropy.

### Strengths
The idea of utilizing some proven models in speech to text generation and migrate these models into brain signal to text generation makes sense. In this area, we've observed quite a bit papers following the same phylosophy.

### Weaknesses
- The experiments are extremely limited, where only one set of experiment reported and the performance is even not close to baseline. Is this results just to illustrate whisper model could reach slightly worse results compared to the baseline? Also, in the original brain-to-text 2025 repo, there are already some wav2vec/whisper similar structures. 

- Utilizing whisper model into brain to text decoding is not new as well. What is the difference between this paper and NeuSpeech paper using MEG data? Merely the data sampling rate and the shape difference won't bring too much novelty.

### Questions
-

### Soundness
2

### Presentation
2

### Contribution
2
