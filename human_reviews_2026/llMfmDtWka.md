# Scaling Speech Tokenizers with Diffusion Autoencoders

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4, 4

## Abstract
Speech tokenizers are foundational to speech language models, yet existing approaches face two major challenges: (1) balancing trade-offs between encoding semantics for understanding and acoustics for reconstruction, and (2) achieving low bit rates and low token rates. We propose Speech Diffusion Tokenizer (SiTok), a diffusion autoencoder that jointly learns semantic-rich representations through supervised learning and enables high-fidelity audio reconstruction with diffusion. We scale SiTok to 1.6B parameters and train it on 2 million hours of speech. Experiments show that SiTok outperforms strong baselines on understanding, reconstruction and generation tasks, at an extremely low token rate of 12.5 Hz and a bit-rate of 200 bits-per-second.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes several improvements to audio tokenizer training. 
Flow matching objective replaces the adversarial training, CTC loss to enforce semantic preservation, decoder finetuning, token CFG.

### Strengths
Auxiliary CTC loss is a good idea and is shown to be very useful.
The flow matching simplifies and speeds up training, and introduces stochastisity to decoding, which might improve audio quality (rather than a deterministic decoder).
Decoder finetuning and Token CFG are intuitive ideas, that show benefits in practice.
Clear writing and multiple ablations.

### Weaknesses
Data is inhouse and not described in detail. Harder to reproduce or generalize to experiments done with different data mixtures.
Results are shown for just one dataset - SeedTTS.

### Questions
Table 5: why are the understanding metrics on Loss R / R+D identical? aren’t you training the encoder/decoder jointly?

Why did you choose to make the encoder causal, and the decoder bidirectional? 

How do you project and predict audio? (e.g. project audio to the encoder, project the noise audio to the decoder, predict the audio)

How about adding the raw audio WER/Sim/UTMOS results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SiTok, a diffusion-based speech tokenizer that achieves high compression ratios while maintaining high-fidelity speech reconstruction. A CTC loss–based semantic regularization method is introduced to enhance the semantic richness of quantized representations. In addition, the paper explores shortcut fine-tuning to accelerate diffusion inference. Extensive experiments demonstrate that SiTok delivers superior reconstruction and downstream understanding performance at low bitrates, and comprehensive ablation studies confirm the effectiveness of each module.

### Strengths
1.**Strong performance at extremely low bitrates.** The proposed SiTok achieves excellent results in both speech reconstruction quality and downstream understanding tasks at very low bitrates. In reconstruction evaluation, SiTok at 200 bps reaches a SIM score of 0.682, with its UTMOS and WER metrics are comparable to those of previous SOTA codec models. On downstream understanding tasks such as LLM based ASR, ER, SV, and KS, SiTok exhibits rich and effective semantic representations.

2. **Comprehensive and well-analyzed ablation studies.** The paper provides extensive ablation experiments that validate the effectiveness of each core component, including model scaling, accelerated diffusion strategies, CTC loss weighting, and reconstruction refinement. The corresponding analyses are clear and insightful.

3. **Exploration of acceleration strategies for diffusion-based tokenizers.** The exploration of shortcut fine-tuning and a light-weight diffusion head offers an solution to accelerate the diffusion decoding process while maintaining high reconstruction quality.

### Weaknesses
1.The overall design of the proposed speech tokenizer lacks novelty. (1) Prior works such as LaDiffCodec [1]
and TadiCodec [2] have already explored diffusion-based decoders to enhance reconstruction quality. (2)
Similarly, models like Baichuan Audio Tokenizer [3] and XY-Tokenizer [4] incorporate ASR-based
supervision to enrich semantic representations after quantization, although they adopt LLM-based
decoders rather than CTC loss.

2.While the paper emphasizes SiTokʼs strong reconstruction quality at high compression rates, it omits
important objective metrics such as STOI and PESQ, which weakens the persuasiveness of the evaluation.

3.Although several strategies for accelerating diffusion inference are explored, the paper lacks concrete
measurements and comparisons of reconstruction speed, such as RTF or related latency metrics.

[1] Yang, Haici, Inseon Jang, and Minje Kim. "Generative de-quantization for neural speech codec via latent
diffusion." ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP). IEEE, 2024

[2] Wang, Yuancheng, et al. "Tadicodec: Text-aware diffusion speech tokenizer for speech language
modeling." arXiv preprint arXiv:2508.16790 (2025).

[3] Li, Tianpeng, et al. "Baichuan-audio: A unified framework for end-to-end speech interaction." arXiv
preprint arXiv:2502.17239 (2025).

[4] Gong, Yitian, et al. "XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech
Codecs." arXiv preprint arXiv:2506.23325 (2025).

### Questions
1.While the paper focuses on reconstruction quality and semantic richness of the speech tokenizer, has the model been evaluated on TTS downstream tasks? Such results could better demonstrate its applicability to
generative scenarios.

2.The reconstructed audio achieves remarkably high SIM scores, even at 200 bps. Could the authors clarify
which component primarily contributes to this improvement? Or does the large-scale training data (2M
hours) play a dominant role in boosting SIM performance?

3.The encoder is causal, whereas the decoder is not. The authors briefly mention this limitation, but I am
curious whether converting the diffusion-based decoder into a causal one would lead to significant
degradation in reconstruction quality, as streaming applications are of growing interest to the community.

Here are some suggestions:

1.The training configuration is insufficiently detailed. It would be helpful to include information such as batch
size and the number of GPUs used.

2.The paper frequently mentions decoder fine-tuning, which seems to correspond to a two-stage training
scheme. The authors are encouraged to clarify and include these details in the training description.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on an important topic in the audio domain—the design of speech tokenizers. By scaling both the model size and the amount of training data, the authors build a highly performant tokenizer and provide a detailed discussion of its training strategies from multiple perspectives.

### Strengths
- The paper scales the speech tokenizer up to a 1.6B-parameter model trained on 2 million hours of speech data, effectively enhancing model performance and achieving state-of-the-art results.

- The authors conduct comprehensive ablation studies on various aspects of the tokenizer’s training process and offer insightful observations on effective training techniques.

### Weaknesses
- The current version appears to lack qualitative analyses or examples of actual reconstructed speech. Since audio reconstruction fidelity is critical for evaluating a speech tokenizer, it would be highly valuable to include qualitative results (e.g., audio samples) to better demonstrate the model’s effectiveness. **If such results are not presented in the rebuttal, I would consider decreasing my score.**

- The main improvements in this work seem to stem from scaling the model and training data, while many methodological components follow prior work. Nevertheless, the systematic analysis and strong pretraining effort provide meaningful progress for the field.

### Questions
It is unclear whether the authors plan to release the model checkpoints or pretrained weights. Without open-sourcing, the paper’s overall impact and reproducibility would be significantly limited. Clarification on this point would be appreciated.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces the Speech Diffusion Tokenizer (SiTok), a diffusion-based autoencoder that jointly optimizes vector quantization and waveform reconstruction. To align its discrete codes with linguistic information, SiTok incorporates semantic regularization via a CTC decoder. To enhance efficiency, the authors apply shortcut fine-tuning and other acceleration techniques to reduce diffusion steps without compromising quality. Trained on 2 million hours of speech with 1.6B parameters, the paper claims that SiTok is effective even at an extremely low token rate (12.5 Hz, 0.2 kbps) and outperforms previous baselines.

### Strengths
- The paper’s strength lies in its simple framework consists of a diffusion autoencoder, vector quantization, and CTC supervision. While not conceptually novel, it’s highly practical. The auxiliary CTC loss proves crucial (WER drops from 33.0 to 4.06, Table 3). Refinements like shortcut fine-tuning and classifier-free guidance further boost efficiency, reflecting thoughtful system design.

- The experiments are comprehensive. Table 5 explores eight design factors (e.g., loss type, codebook settings, frame rate) and evaluates both reconstruction (WER, SIM, UTMOS) and understanding (ER, SV, KS, ASR) tasks.

- At 0.2 kbps (12.5 Hz), SiTok delivers competitive performance in a highly compressed regime. With a single codebook, it achieves WER 4.06 and SIM 0.641

### Weaknesses
- The paper prefers mel-spectrograms over raw waveforms without strong justification. Modern architectures can handle raw waveforms efficiently, and prior speech tokenizers (e.g., EnCodec, WavTokenizer, BigCodec, Mimi) operate directly on them successfully. Claims that waveforms are inefficient or adversarially complex are unconvincing, especially given that adversarial methods have been applied effectively in prior work. Mel-spectrograms are lossy and require a separate vocoder (Vocos), yet no comparison to waveform-based models is provided.

- The method relies on a single VQ codebook, despite evidence (Table 5) that residual VQ (RVQ) substantially improves performance (WER 4.06 → 2.50 with 8 codebooks). Leading tokenizers employ RVQ to avoid bottlenecks at low token rates. The paper provides no rationale for using a single codebook, ignores alternatives such as FSQ.

- Diffusion-based speech tokenization is not novel; prior work includes CosyVoice, Vevo, FireRedTTS, NaturalSpeech 3, DiTAR, and OZSpeech. The claimed novelty of end-to-end training with CTC supervision is not clearly superior to established two-stage approaches.

- The regression loss achieves a WER of 4.66 compared to 4.06 for diffusion, so the improvement is far from marginal. This raises questions about the practical benefit of using diffusion loss over a simpler L1 loss.

- The paper is titled "SCALING Speech Tokenizers" but Table 4 shows scaling doesn't work: the 1.61B (XL) model performs worse than the 1.12B (L) model on key metrics (ASR: 5.07 vs 4.95; SV: 14.7 vs 13.8).

### Questions
- What is the real-time factor (RTF) and actual inference latency in milliseconds for the method compared to single-pass baselines?

- How does task performance change if the CTC decoder is scaled proportionally (e.g., 24 layers for the XL model) rather than keeping it fixed at 4 layers?

- Was Finite Scalar Quantization (FSQ) evaluated as an alternative to vector quantization (VQ)?

- Can statistical significance tests be provided for the reported performance gap, given that the improvement is minimal?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Speech Diffusion Tokenizer (SiTok), a diffusion autoencoder-based speech tokenizer. It aims to achieve extreme compression, high-quality reconstruction, and effective semantic representations for speech language modeling simultaneously. SiTok addresses the limitations of existing speech tokenizers in balancing compression rate, reconstruction quality, and semantic representation by jointly learning semantic-rich representations and high-fidelity audio reconstruction.

### Strengths
1. Performance: SiTok achieves superior performance in both speech reconstruction and understanding tasks at extremely low bit rates (0.2kbps) and token rates (12.5Hz), outperforming several strong baselines. This highlights its potential for efficient speech representation and modeling.
2. Innovative Design: The combination of diffusion models and semantic regularization is a novel approach in speech tokenization. These innovations enable SiTok to learn representations that are both acoustically faithful and semantically meaningful.
3. Flexibility: The model's performance can be further enhanced by adjusting parameters such as model size and codebook configurations, offering flexibility for different applications.

### Weaknesses
1. Lack of Direct TTS Task Evaluation: While the paper demonstrates strong performance on various speech understanding tasks such as ASR, ER, KS, and SV, it does not directly evaluate the proposed SiTok model on TTS tasks. This is a significant gap, as TTS tasks have unique requirements and challenges that may not be fully addressed by the current evaluation metrics.
2. Lack of Detail on Multi-Codebook CTC Decoder Implementation: The paper does not provide specific details on which layer the CTC decoders are applied to in the multi-codebook setup. This lack of clarity can make it difficult for other researchers to reproduce and build upon the work.
3. Real-time Challenges: Despite efforts to accelerate decoding, diffusion models' iterative sampling steps may still pose latency issues for real-time applications. Further optimization is needed to improve real-time performance.
4. Resource Intensity: The large model size (1.6B parameters) requires substantial computational and storage resources for training and inference, potentially restricting its deployment on resource-constrained devices.

### Questions
1. Why is there no demo page?
2. Ablation study on CTC loss supervision:
  The ablation study shows significant improvements in both WER and similarity metrics when CTC loss supervision is applied. While the improvement in WER is understandable as it directly relates to the semantic alignment of the speech tokens with the text, the substantial increase in similarity is puzzling. Typically, semantic and acoustic aspects involve a trade-off, and enhancing one might come at the cost of the other. Could the authors provide more detailed experimental insights into why the similarity metric shows such a marked improvement? This would help in better understanding the model's behavior and the interplay between semantic and acoustic features.
3. Comparison with Tokenizers Trained for Understanding Tasks:
  Many of the tokenizers compared in the understanding tasks seem to be trained primarily for reconstruction tasks. The performance metrics on understanding tasks may not be as convincing as they could be. Could the authors provide additional comparative experiments with tokenizers that are specifically trained for understanding tasks, such as S3Tokenizer? This would help in assessing the true effectiveness of SiTok in the context of speech understanding.

### Soundness
2

### Presentation
3

### Contribution
3
