# Whisfusion: Parallel ASR Decoding via a Diffusion Transformer

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 10, 4, 2, 0

## Abstract
Fast automatic speech recognition (ASR) is crucial for applications such as captioning and transcription. Although modern ASR encoders can process up to ~30 seconds of audio in a single pass, Whisper-style autoregressive (AR) decoders still generate tokens sequentially, making decoding latency grow linearly with utterance length. We propose Whisfusion, a non-autoregressive (NAR) ASR framework that fuses a frozen pre-trained Whisper encoder with a masked-diffusion text decoder. At each diffusion step, the decoder conditions on the full acoustic context and updates all tokens in parallel, mitigating the AR latency bottleneck while preserving Whisper-compatible generative structure. A lightweight cross-attention adapter trained via parameter-efficient fine-tuning bridges audio and text, and we introduce Parallel Diffusion Decoding (PDD), an ASR-tailored batch-parallel sampling scheme that improves the accuracy–latency trade-off in low-to-mid batch regimes. With 6.5k hours of training data, Whisfusion reaches 4.9\% WER on LibriSpeech test-clean, comparable to similarly sized Whisper model (Whisper-small at 5.0\%), while enabling much faster decoding. In particular, on 20–30s segments within Whisper’s 30s window, Whisfusion reduces decoding time from 674.7 ms to 80.7 ms (8.4× faster) at similar accuracy, demonstrating an efficient NAR operating point for Whisper-compatible ASR.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Whisfusion, a non-autoregressive (NAR) automatic speech recognition (ASR) framework that fuses a pre-trained Whisper encoder with a masked diffusion text decoder via a lightweight cross-attention adapter trained using parameter-efficient fine-tuning (PEFT). The method improves the decoding speed of traditional autoregressive (AR) decoders by enabling parallel generation across the full acoustic context. A new Parallel Diffusion Decoding (PDD) strategy enables multi-candidate iterative refinement with negligible speed loss. 
Fine-tuned solely on LibriSpeech (960h), Whisfusion achieves a lower WER than Whisper-tiny, and offers comparable latency on short audio. For longer utterances (>20s), it is up to 2.6× faster than the AR baseline.

### Strengths
- The paper is generally well-written. 
- Improving the decoding speed of traditional AR decoders, by incorporating a masked diffusion text decoder is interesting.
- Using the cross-attention adapter trained via parameter-efficient fine-tuning (PEFT) to bridge the encoder and decoder is new.

### Weaknesses
- The paper does not correctly distinguish the decoding latency and the decoding speed. NAR decoding improves the decoding speed, but not the decoding latency. The encoder consumes the whole utterance, and then the NAR decoder outputs the entire text sequence and iteratively refine it (all positions in parallel).
- The latency of NAR decoding is a disadvantage, not suitable for streaming decoding. Streaming ASR performance is not evaluated for the proposed method.
- Whisfusion still lags behind Whisper-small (comparable in size) in WER, suggesting the approach has not yet closed the accuracy gap.
Compared to prior results trained solely on LibriSpeech (960h), the results of Whisfusion also lag behind.
https://github.com/thu-spmi/ASR-Benchmarks/blob/main/README.md#Librispeech
- Limited dataset scope. Results are confined to a LibriSpeech (clean English).
- The length estimation is a key challenge for NAR decoding. It is good to see the analysis of length estimation (but in appendix). The details of how Whisfusion performs length estimation are not clear.

### Questions
see above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
5

### Summary
Authors present strategy for finetuning whisper models with diffusion for ASR tasks. This is achieved with two stage training strategy with pretrained models. In addition, they present a parallel decoding strategy that takes advantage of diffusion based inference for additional gains.

### Strengths
Architecture is suitably novel that it contributes to ASR research, and the use of diffusion models in lieu of autoregression is an active field of discussion for general ML research. Given the acceptable performance on a common benchmark, these two aspects alone merit acceptance.

### Weaknesses
Proposed model uses pretrained checkpoints for SFT while autoregressive models can be trained from scratch. It would be useful if authors can present information on full pipeline training with current model instead of use of pretrained checkpoints.

Training was performed on a relatively easy benchmark (LS is read speech from audio-book actors). This makes it unknown how well diffusion approaches work for real-world speech recognition cases (telephony, conversational speech, high noise presence).

### Questions
Requests from authors for camera ready submission:
- If possible, please provide additional evaluation numbers on non-librispeech benchmarks. No need to train whole model, but having deltas from CommonVoice, Fisher, and the like would help show how model responds to domain drift. These can be provided in appendix.
- Move discussion of diffusion loss from appendix to main body of work. Diffusion is less common in speech recognition and readers would benefit from review.
- Resize image on page 20, it's not even close to readable as currently presented.
- Was there an attempt to train from scratch? Even if results are paltry, there would be community benefit to acknowledge if there are limitations on diffusion in full pipeline training.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Whisfusion, a novel non-autoregressive (NAR) ASR framework that combines a pre-trained Whisper encoder with a diffusion-based text decoder. The two components are connected via a lightweight cross-attention adapter trained through parameter-efficient fine-tuning . The authors also design a batch-parallel multi-step decoding strategy that improves recognition accuracy while maintaining efficiency. Experiments on LibriSpeech (960h) show that Whisfusion outperforms Whisper-tiny and achieves up to 2.6× faster decoding speed for long-form utterances.

### Strengths
Novel architecture: Using a diffusion transformer as a decoder for ASR is highly original and demonstrates an interesting fusion of generative modeling and speech recognition.

Effective integration: The cross-attention adapter provides a clear and efficient way to connect the Whisper encoder and the diffusion decoder.

Improved decoding strategy: The proposed multi-step batch decoding (similar to beam search) is a well-motivated idea that improves the quality-speed trade-off compared to previous NAR models that rely solely on argmax decoding.

Clear experimental gains: On LibriSpeech, Whisfusion achieves measurable WER improvements over Whisper-tiny, showing the potential of diffusion-based decoding in ASR.

### Weaknesses
Limited performance improvement: Although Whisfusion uses the Whisper-small encoder, its performance still lags significantly behind the Whisper-small model, raising questions about the effectiveness of the diffusion-based decoding in capturing linguistic dependencies.

Lack of text length modeling: The model does not explicitly predict or control text length, which could lead to substantial computational waste, especially for short utterances.

Limited comparisons: The paper does not compare with other recent non-autoregressive ASR baselines , making it difficult to contextualize the contribution.

Single dataset evaluation: All experiments are conducted only on LibriSpeech; results on other datasets (e.g., TED-LIUM, CommonVoice) would make the conclusions more convincing.

### Questions
Have you considered using adaptive text length prediction to reduce unnecessary computation?

Could Whisfusion benefit from fine-tuning both encoder and adapter jointly on downstream datasets?

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
5

### Summary
A text diffusion decoder is extended by cross-attention to an audio encoder, to build a non-autoregressive speech recognition model using diffusion.

The motivation is to improve the recognition speed.

The authors use the pretrained Whisper encoder and leave it frozen through all the experiments. The authors use a pretrained  mdm-170M diffusion LM.

There are two training stages: In the first stage, only the added cross-attention part is trained. In the second stage, the whole decoder is trained.

Librispeech is used for training (finetuning) and also for evaluation. As the authors use pretrained models, a lot more data was implicitly used.

Their final Whisfusion model performs better than Whisper-tiny, but worse than all other Whisper variants. All the Whisper variants were taken as-is and not fine-tuned on Librispeech.

### Strengths
* Using diffusion for ASR is an interesting and promising direction.
* Good speedups for recognition.
* Study on the parallel diffusion decoding (PDD) is interesting and shows its importance.

### Weaknesses
* The experimental setting is bad. We cannot really learn much about the most relevant questions (how does such a diffusion model compare to other alternative ASR models, under the same conditions, same training data that was used implicitly or explicitly).
* Phrasing it as novel is not totally correct.
* It is sold as a good solution specifically for long-form ASR, but then the experiments show that this is very it performs really bad.
* No real scaling laws analysis.
* Analysis should be extended.

See comments/questions below for more details on these points.

### Questions
Abstract " We propose Whisfusion, the first framework to fuse a pre-trained Whisper encoder with a text diffusion decoder." - I'm not sure what "first" refers to. This very specific thing, that a pre-trained Whisper encoder was used specifically? Or the more generic aspect that pretrained models were used to get a diffusion ASR model? Or does this say that this is the first diffusion ASR model? The latter is definitely not true. For example, there is:
- TransFusion: Transcribing Speech with Multinomial Diffusion, https://arxiv.org/abs/2210.07677
- Cross-Modality Diffusion Modeling and Sampling for Speech Recognition, https://www.isca-archive.org/interspeech_2024/yeh24_interspeech.pdf
- Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing, https://arxiv.org/abs/2509.16622 (to be fair, very recent...)
- Drax: Speech Recognition with Discrete Flow Matching, https://arxiv.org/abs/2510.04162 (ok, to be fair, very recent. but check its related work)

Abstract "Fine-tuned solely on LibriSpeech (960h), Whisfusion achieves a lower WER than Whisper-tiny (8.3% vs. 9.7%)" - these numbers are not at all impressive... There are much simpler pure CTC models (i.e. also non-autoregressive, but even much faster than diffusion) with much better numbers.

"Novel NAR Framework" - I think describing it as novel is overselling it. A couple of similar models have been proposed before. See above.


No good comparison on top of Whisper: E.g finetune Whisper on Librispeech. Or finetune a CTC on top of the Whisper encoder on Librispeech. Or finetune another smaller/simpler decoder on top of the Whispr encoder on Librispeech. Or finetune some LLM on top of the Whisper encoder.


"batch-parallel, multi-step decoding ... minimal impact on speed" - I assume this is only the case because you parallelize it, assuming you have enough cores idling around? Let's say you already fully utilize the GPU before (e.g. because of batched decoding multiple audio files in parallel), then this would have a big impact on speed?


"... establishing a new, efficient operating point for long-form ASR"  - but is it actually performing bad for longer audio? (Table 3)

Table 3: Usually you mark the best results in bold. Here you just mark Whisfusion results always in bold, which is thus misleading.

Multiple different model sizes should be tested, for some scaling law analysis.

Model performs bad on long-form. Why? Any analysis? Maybe could be better with better decoding? But then again would probably be slower.


What do we really learn from this? Just that it works at all? But that was to be expected, and that was also already shown before. What else? The WER comparison to other models would be most interesting, but in its current form (Table 3, Table 4), it is somewhat meaningless, as the training data in each case is different.

Sec 5.2.1 "”w/o Acoustic Conditioning1“" - what is "Conditioning1"? Typo?

Sec 5.2.1 / Table 5: I don't quite understand the w/o acoustic conditioning experiment: In what stage do you remove this? Stage 1? Stage 2? Both stages? Do you even do both stages here, or only stage 1? I assume you remove it for both stages, and then also don't use it in recognition? I assume you anyway did both training stages? So basically you finetuned the diffusion LM just as a LM without conditioning? "Despite masking only 30% of the tokens from the ground truth transcript" - what do you mean by that? You mask 30%, and wanted to see how good the model can recover it, just as a LM, without conditioning? Again, to be clear: Here, you also trained the model without conditioning? So then the result (150.8% WER) is unexpected, right? You would maybe expect 30% or less. Either I don't fully understand the experiment, or sth is broken here. Did you analyze this? What does the model generate as output?


"we .. propose Parallel Diffusion Decoding (PDD), a novel inference strategy ..." - It's not really so novel. Similar ideas have been done for diffusion models before. For example (mostly citing only the recent ones, but there are many older ones as well):
- https://arxiv.org/abs/2306.17775
- https://arxiv.org/abs/2503.02039
- https://arxiv.org/abs/2505.24857
- https://arxiv.org/abs/2508.08712 (it's a survey, it cites many earlier works)
- https://arxiv.org/abs/2509.25188
- https://arxiv.org/abs/2509.23146
- https://arxiv.org/abs/2510.21961 (ok, to be fair, that's very recent, but see their related work)


There also should be experiments for not importing pretrained models, for comparison.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 5

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces Whisfusion, a non-autoregressive (NAR) Automatic Speech Recognition (ASR) framework designed to overcome the latency bottlenecks of autoregressive models like Whisper. The model fuses a frozen, pre-trained Whisper encoder with a pre-trained text diffusion decoder Masked Diffusion Model (MDM) using a lightweight cross-attention adapter. To train this hybrid architecture effectively, the authors employ a 2-stage curriculum: first training only the adapter, then fine-tuning the decoder with a high masking ratio to specialize it for ASR. They also propose Parallel Diffusion Decoding (PDD), an inference strategy that generates multiple candidates in parallel and iteratively refines them, selecting the best one based on confidence scores.

### Strengths
- NAR Framework: the first architecture to fuse a Whisper encoder with a text diffusion decoder for ASR5.
- Parallel Diffusion Decoding (PDD): A batch-parallel, multi-step decoding strategy that improves accuracy by increasing candidate count ($k$) with negligible latency impact
- Speed-Accuracy Operating Point: Achieves lower WER than Whisper-tiny (8.3% vs 9.7% on LibriSpeech test-clean) while being significantly faster (up to 2.6x) on long-form audio.

### Weaknesses
- Inadequate Baselines for Speed Claims: The paper positions the model as a high-speed alternative to autoregressive (AR) decoding but only compares it against AR models (Whisper variants). It fails to compare against established non-autoregressive or limited-context architectures (e.g., CTC-based models, Transducers with greedy decoding) that are inherently fast. Without these baselines, the claim of a "superior speed-accuracy trade-off" is unsubstantiated, as a standard 300M-parameter CTC model with greedy decoding might achieve similar or better performance without the complexity of iterative diffusion.

- Uncompetitive Accuracy for Model Size: Despite using a ~301M parameter model (larger than Whisper-small's 244M), it achieves significantly worse accuracy across different length ranges (Table 3 WER column). An 8.3% WER on test-clean is notably high for a modern ASR model of this size, suggesting the non-autoregressive trade-off may be too severe for practical use compared to current state-of-the-art.

- Failure on Long-Form Audio: A key selling point is the constant inference speed on long audio (up to 30s). However, the paper admits severe accuracy degradation on 20-30s segments (rising to 15.9% WER due to training data scarcity). This negates the practical value of the speed advantage on long-form audio if the resulting transcriptions are unreliable.

- Questionable Utility of the Diffusion Decoder: The ablation study reveals that without the acoustic cross-attention adapter, WER spikes to ~150% (Table 5), indicating the pre-trained textual knowledge of the MDM is insufficient on its own. This raises doubts about whether the complex, iterative diffusion decoder is adding necessary value, or if a simpler, single-step non-autoregressive decoder conditioned on the same strong Whisper encoder embeddings would perform equally well.

### Questions
I like to hear authors feedback on the weakness points raised.

### Soundness
2

### Presentation
2

### Contribution
2
