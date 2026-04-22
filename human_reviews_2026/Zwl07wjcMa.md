# Sylber 2.0: A Universal Syllable Embedding

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
Scaling spoken language modeling requires speech tokens that are both efficient and universal. Recent work has proposed syllables as promising speech tokens at low temporal resolution, but existing models are constrained to English and fail to capture sufficient acoustic detail. To address this, we present Sylber 2.0, a universal framework for coding speech at the syllable level, enabling efficient temporal compression and high-fidelity reconstruction across multiple languages and expressive styles. Building on the original Sylber, Sylber 2.0 improves both linguistic coverage and reconstruction quality by training on diverse multilingual speech and introducing a syllable-level acoustic encoder and vocoder. Sylber 2.0 achieves a very low token frequency around 5 Hz, while retaining both linguistic and acoustic detail. Experiments show that it performs on par with previous models operating on high-frequency baselines, and it outperforms the original Sylber by a significant margin. We further demonstrate the efficacy of Sylber 2.0 in downstream tasks, especially in English TTS and low-resource ASR. Sylber 2.0 based TTS model, SylFlow, can generate speech with competitive intelligibility and quality with current SOTA models using only 72M, and be more effective in resource-constrained ASR than previous speech coding frameworks. In sum, we establish an effective syllable-level abstraction for general spoken language.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Sylber 2.0, a universal framework designed to encode speech at the syllable level across multiple languages. The goal is to create speech tokens that are both efficient (low temporal resolution) and universal. The framework consists of a content encoder for linguistic information, an acoustic encoder for vocal details, and a boundary detector to identify syllable breaks. A lightweight vocoder then reconstructs high-fidelity audio from these compressed embeddings. The model is trained on diverse multilingual data without using any text.

### Strengths
Originality: The paper is original in that it attempts to create a universal framework that learns syllabic structure directly from audio across 102 languages without any textual supervision. This moves significantly beyond previous models that were often constrained to English. The proposed architecture is also novel 
Quality: The paper performed comprehensive evaluations on both reconstruction and a downstream TTS task. The authors also provided visualization to show that the model has learned consistent, syllable-like segments across different languages.
Clarity: The writing is easy to follow and the visual aids were provided.
Significance: The paper makes significant contribution in achieving a compression of speech at an average of 4.8 Hz, reducing the computational cost and memory requirements for training and running large-scale spoken language models, making them more scalable and efficient. The fact that the model works across 102 languages is also rather significant as it could serve as a powerful foundation for future multilingual speech understanding and generation systems, especially in low-resource languages.

### Weaknesses
1. The assessment on downstream task is rather lacking. The authors showed the performance on one TTS dataset where the proposed method's performance is not that competitive on WER. More tasks and datasets should be included to provide a better sense of the method's quality.
2. The syllable detection performance is not that high, especially the precision. Suggesting the model might be over-segmenting the speech?
3. Ablation studies on the different architecture improvement over Sylber would be nice

### Questions
1. Is the model learning linguistically meaningful syllables? 
2. Could the authors perform more evaluations on different downstream tasks? Especially for different languages to showcase the multilinguality?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Sylber 2.0, a framework for universal speech tokenization. The primary goal is to create a speech representation that is both highly efficient (targeting a ~5 Hz syllabic rate) and high-fidelity, overcoming the limitations of prior work like the original Sylber, which was constrained to English and lacked acoustic detail.

The core methodological novelty is a new, disentangled token structure composed of three parts: duration (d), linguistic content (C), and acoustic information (A). This is enabled by three modules that include:
- A syllable-guided acoustic encoder (a CNN + Transformer stack) that runs in parallel to the content encoder, capturing speaker and style information.
- A trained boundary detector that replaces a slower, iterative segmentation algorithm, enabling faster, parallel processing.
- A within-segment positional encoding (wSegPE), a method to help the vocoder reconstruct audio from variable-length syllable segments.

The authors claim this new framework can compress diverse, multilingual speech (across 102 languages) to an average of 4.8 Hz while enabling high-fidelity reconstruction that substantially outperforms the original Sylber and remains competitive with high-frequency tokenizers.

### Strengths
- The goal of creating a universal, high-fidelity, and highly compressed speech token is a critical and high-impact challenge for the spoken language modeling community. Success here would enable models to process much longer speech contexts efficiently.

- The paper introduces several intelligent and specific methodological ideas. The central concept of a disentangled (d, C, A) token is elegant. The syllable-guided acoustic encoder is a specific, new architecture designed to solve the well-known problem of acoustic information being "marginalized out" by self-supervised content encoders. Furthermore, the wSegPE is a clever technical solution to the inherent conflict between variable-length syllabic tokens and fixed-rate vocoders.

### Weaknesses
- The paper lacks empirical validation for its core contribution. The acoustic encoder ('A' token) is presented as the key innovation for achieving high-fidelity reconstruction. However, the paper provides no ablation studies to prove its impact. The main comparison in Table 2 is against the original "Sylber," which is not an apples-to-apples comparison. The gains are confounded by multiple variables: a new (multilingual) dataset, a new (and likely better) vocoder, and the removal of silent masking. It is difficult to determine if the novel acoustic encoder contributed significantly, or if the gains are simply from the new dataset and vocoder.

- The paper's core SSL methods, frame-wise self-distillation and self-segmentation distillation, are explicitly borrowed from prior work. The novelty lies in the new architecture (the (A) encoder) and the boundary detector (wSegPE).

### Questions
- To properly evaluate the impact of your individual methodological contributions, could you provide an ablation study where you isolate each architectural novelty? For instance, starting from a common baseline (e.g., Sylber 1.0 + new dataset).
- Regarding the high 3.29 WER in the TTS task (Table 3): Do you believe this is a limitation of your small SylFlow model, or does it suggest that your syllable-level token, by its very nature, creates a "fidelity ceiling" by smoothing essential, sub-syllabic phonetic details?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Sylber 2.0, a syllable-level speech coding model, which is built upon the original Sylber. By introducing syllable-level acoustic embedding and vocoder, Sylber 2.0 enables better linguistic coverage and higher reconstruction quality than Sylber. Experiments on speech reconstruction show that it can achieve performance comparable to previous higher-frequency speech tokenizers. Zero-shot TTS experiments indicate that it can obtain result comparable to other TTS models, with much less training resources.

### Strengths
1) The proposed Sylber 2.0 achieves an impressive speech compression rate with token frequency of around 5Hz on many languages, while retaining both linguistic and acoustic details. Extensive experiments and analysis manifest the effectiveness of this method. 
2) This paper provides detailed implementation description, and presents good academic expression, data visualization, and result analysis.

### Weaknesses
1) The overall novelty is limited. This work extends the original Sylber system by adding the acoustic encoder and vocoder, and also improves the training process of content encoder. These changes are incremental. 
2) This paper doesn't report quantitative ablation results of the proposed changes to the original Sylber.

### Questions
1) In Table 3, how about the TTS results of the original Sylber? 
2) This system is claimed to compress speech into syllable-level embedding. Should it demonstrate greater benefits in terms of the WER metric for resynthesis and TTS? 
3) Though Table 6 reports the RTF of extracting the content embeddings, as the system looks complicated, I am also concerned about overall efficiency of this system when compared to other speech encoding/tokenizer models and also the original Sylber.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Sylber 2.0, a universal framework for encoding speech into syllable-level embeddings at ~5 Hz frequency. The key innovation is extending syllable-based speech tokenization from English-only to 102+ languages while significantly improving reconstruction quality. The system learns syllable segmentation through self-supervised learning without text supervision, representing each syllable with three components: content embeddings (linguistic information), acoustic embeddings (speaker/style), and duration tokens. A lightweight vocoder reconstructs 24 kHz audio from these compressed representations. The authors demonstrate near-lossless compression performance comparable to high-frequency baselines (86 Hz) and showcase practical utility by training a small 72M parameter TTS model (SylFlow) that matches SOTA models 5-10× larger.

### Strengths
1. Achieves lowest reported token frequency (4.8 Hz average) for multilingual speech, dramatically reducing computational costs for downstream modeling compared to existing methods (12.5-86 Hz)

2.  Successfully learns syllabic structure across 102 languages without text supervision, addressing the major limitation of prior work (Sylber 1.0) which only handled English

### Weaknesses
1. **Insufficient downstream task validation**: The paper proposes a new speech representation primarily motivated by speech language modeling, yet only a small portion (lines 462-476) demonstrates its usage. Only TTS results are shown, which is insufficient since TTS is relatively simple and can be trained effectively with Mel-Spectrogram + Vocoder without any tokenization. The paper needs to justify the benefits of the proposed embedding for more diverse downstream tasks and identify which tasks would actually benefit from syllable-level representations

2. **Overclaimed TTS efficiency benefits**: The claim that the proposed embedding enables training smaller models with less data for TTS is overclaimed. It is well-established that good TTS models can be trained on small datasets like LJSpeech (40h) using Tacotron2 by overfitting to a single speaker, so requiring fewer parameters for smaller datasets is expected. The evaluation should be conducted on standard benchmarks like seed-tts-eval and include comparisons with recent baselines such as SparkTTS (https://github.com/SparkAudio/Spark-TTS) to show it can modeling diverse speaking style, speakers, languages, etc.

3. **Unclear advantage over model-free representations**: For multilingual speech representation, mel-spectrograms provide lossless, model-free representations. The fundamental issue with model-based compressors is generalization. The paper does not adequately justify why syllable embeddings should be preferred over mel-spectrograms for TTS, especially when an I-STFT based vocoder (e.g., https://github.com/gemelo-ai/vocos) can transform mel-spectrograms back to audio without requiring learned compression

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
2
