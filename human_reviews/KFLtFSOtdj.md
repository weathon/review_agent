# Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models

- Decision: Reject
- Scores: 6, 1, 5, 6, 5

## Abstract
Building upon advancements in Large Language Models (LLMs), the field of audio processing has seen increased interest in training audio generation tasks with discrete audio token sequences. However, directly discretizing audio by
neural audio codecs often results in sequences that fundamentally differ from text sequences. Unlike text, where text token sequences are deterministic, discrete audio tokens can exhibit significant variability based on contextual factors, while still producing perceptually identical audio segments. We refer to this phenomenon as \textbf{Discrete Representation Inconsistency (DRI)}. This inconsistency can lead to a single audio segment being represented by multiple divergent sequences, which creates confusion in neural codec language models and results in omissions and repetitions during speech generation. In this paper, we quantitatively analyze the DRI phenomenon within popular audio tokenizers such as EnCodec. Our approach effectively mitigates the DRI phenomenon of the neural audio codec. Furthermore, extensive experiments on the neural codec language model over LibriTTS and large-scale MLS datasets (44,000 hours) demonstrate the effectiveness and generality of our method. The demo of audio samples is available online~\footnote{\url{https://consistencyinneuralcodec.github.io}}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses the Discrete Representation Inconsistency (DRI) phenomenon in neural audio codecs, where identical audio segments are represented by different discrete token sequences, reducing downstream task performance on speech generation. The authors propose two methods, slice-consistency and perturbation-consistency, to mitigate DRI by aligning token representations from audio slices and perturbed audio. They show significant consistency improvements in popular neural audio codecs and reduced word error rates (WER) in speech generation tasks using large datasets.

### Strengths
- The authors identify an interesting problem with audio tokenizers where audio, with or without contextual information,
is encoded by the audio tokenizer into different audio tokens unlike text tokenizer which decode to identical text tokens
- The proposed method aims to retain the original receptive field while enabling the model to address the trade-offs between audio reconstruction quality and resolving the many-to-one problem. To achieve this the method uses a segment of audio randomly sliced, and the encoded representation from this sliced segment is required to closely approximate the corresponding representation obtained from the entire audio. To address many-to-one problem, the representation of an audio and its representation after applying slight spectral
perturbation are forced to be aligned. 
- The experiments cover various neural audio codecs and datasets across different bandwidth settings and model sizes, supporting the generality of the proposed solution. The results demonstrate large improvements in consistency metrics (up to 36.29% in deeper layers), WER reduction (3.72%), and speaker similarity increase (5.68%)

### Weaknesses
- The paper mainly focuses on consistency-based methods and doesn't extensively compare these approaches to other potential techniques for handling the many-to-one mapping problem, such as the simpler methods introduced in the introduction section (e.g., setting the kernel size to 1). Including such baselines would be helpful for comparison in Table 2.
- The paper would benefit from a baseline without the consistency constraint. The current baselines are different methods/codecs like Encodec, DAC, etc., which might be trained on different datasets. This raises the possibility that the better reconstruction in the author's proposed method might be due to larger training data, rather than the addition of consistency-based loss. To verify this, it would be helpful to include a baseline RVQ-GAN or a model without the additional losses.

### Questions
- Can the authors include the two baselines suggested in the weaknesses section? The first is a kernel size of 1, and the second is the baseline performance of a model trained on the same data without consistency loss. If this isn't possible or necessary, could the authors provide a justification?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper investigated the problem of Discrete Representation Inconsistency (DRI), where a single audio segment is represented by multiple divergent token sequences. This causes confusion in neural codec language models and leads to omissions and repetitions in speech generation. The authors proposed several consistency regularization techniques that enhance codec reconstruction quality to address this problem. The authors also show that text-to-speech (TTS) systems trained on Encodec using these proposed methods achieve better performance.

### Strengths
The problem is interesting.

### Weaknesses
1. There is no theoretical evidence presented to support that improved consistency accuracy in codec models leads to better performance in codec and downstream audio LLM tasks. It would be valuable to see further analysis on this aspect. While the authors highlight the consistency improvement achieved by the proposed method, both the evaluation and training methods (i.e., slice method) are consistent. Therefore, the observed improvement is expected.
2. Neural codec language models for music and sound effects generation (e.g., AudioGen, MusicGen) are also highly relevant. Showing the proposed method used for codec and language models on music and sound effect benchmarks and comparing them with state-of-the-art models is essential to establish the effectiveness of the approach.
3. The reported evaluation of TTS models appears inconsistent with existing literature, particularly regarding WER results. It seems likely that re-ranking may have been used before the WER evaluation, where multiple samples are generated, and the one with the best WER is selected. For instance, VALL-E, known for its limitations in generating speech precisely aligned with text transcriptions, typically achieves around 10% WER on LibriSpeech when trained on LibriLight, which can be improved to approximately 5% through re-ranking. In my opinion, the 1.37% WER reported in the paper is a groundbreaking achievement for TTS. While I recognize that the proposed method could enhance TTS robustness, such a substantial improvement is surprising without further modifications to the training pipeline or model architecture. Additionally, VoiceCraft generally achieves around 20% WER on LibriTTS without re-ranking, whereas the results in the paper show a considerably lower WER of about 2-3%. The paper does not specify how TTS samples were generated; if re-ranking was applied, please clarify this and provide results without re-ranking for an accurate comparison.
4. Although the studied question is interesting, its scientific contribution is limited, or at least needs to be supported by more experiments and results.

### Questions
1. Please clarify the use of re-ranking methods for TTS evaluation. If re-ranking was used, the authors should report performance across different re-ranking levels (e.g., 5, 10, 20 samples, or no re-ranking) to provide a more comprehensive view of its impact on results.

2. To comprehensively assess the TTS system's robustness, I recommend conducting additional evaluations using challenging sentences, such as "22222222 hello 22222222" (some examples can be found in https://ralle-demo.github.io/RALL-E/ or any reasonable test set is acceptable).

3. The baseline codec models in this work are trained on short utterances (around 1 second). It remains unclear whether training a codec model on longer utterances could improve consistency accuracy. Intuitively, codec models trained on short clips may be more susceptible to acoustic variance (the semantic context to leverage is limited). An additional useful experiment would be to include the Mimi codec (https://github.com/kyutai-labs/moshi) as a baseline, given its training on 12-second speech clips and a 7 million-hour dataset. This comparison could offer valuable insights into the effect of training duration and codec capability on consistency accuracy.

4. I recommend adding more perturbations, such as non-audible noise, reverb, or slight volume augmentation when measuring consistency accuracy. This would help in evaluating the robustness of the model's consistency accuracy under varied conditions.

5.  Please include a breakdown of the WER metrics, specifying insertion, deletion, and replacement errors. As the aim of the proposed method is to address omissions and repetitions during speech generation, a detailed report on these error types is crucial to understanding the system's performance.

6. Further evaluation is recommended on music and sound effects benchmarks to validate the effectiveness of the proposed method, as suggested in the weakness part.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper is about analysis and solve the problem of Discrete Representation Inconsistency.

### Strengths
1. The paper is well-written and easy to follow.

2. The performance is comparative.

3. The Discrete Representation Inconsistency seems an important question.

### Weaknesses
1. Fair Comparison: The authors expanded the training dataset to 44,000 hours. However, in Table 1, I do not think other models, including EnCodec and DAC, were trained on the same data. Is this fair for the results in Table 1?
2. Unprofessional Expressions: The format of the abstracts violates the conference rules. In Figure 3, the word "perception" is presented in both italics and normal text. The equations are not tagged with numbers like "(1), (2)." There is no underline in Table 1, Column "ViSQOL." The bold "Ours and Ours w/o consistency constraint" in the caption of Table 2 should be replaced with quotes. This raises serious questions about the paper, as there is still a long way to go.
3. Efficiency: How large is the model? Is there any insight regarding the parameters and FLOPS? Could the performance boost come from a larger model?
4. Implementation Details: Some important details are not mentioned. The variables related to $\lambda$ in the equation in line 285 are not further detailed in this paper. Are they all set to 1?
5. Novelty: The novelty of this paper is rather weak. There is no structural improvement, only further utilization of the method in Phaseaug. However, the analysis of Discrete Representation Inconsistency is interesting. I consider this as a minor weakness.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper  introduces and analyzes the Discrete Representation Inconsistency (DRI) phenomenon, which occurs in discrete audio tokens. DRI leads to confusion in large language models (LLMs) during prediction, causing omissions and repetitions in speech generation. The inconsistency arises because the same audio segment can be encoded into different discrete sequences depending on the context, complicating the model’s prediction process.

To address this issue, the authors propose two solutions:

Slice-Consistency Method: This method ensures that a randomly sliced segment of audio has an encoded representation consistent with that of the entire audio. It helps maintain consistency without reducing the model's ability to understand the audio.

Perturbation-Consistency Method: This method aligns the encoded representations of audio before and after slight spectral perturbations, enhancing the model's robustness. The perturbations are subtle enough to be imperceptible to human hearing but help the model remain stable against small variations.

The authors conduct extensive experiments to validate the effectiveness of these methods. The results demonstrate that these approaches significantly alleviate the DRI phenomenon and improve speech generation performance. Metrics such as reduced Word Error Rate (WER), increased speaker similarity, and enhanced naturalness of generated speech confirm the success of the proposed solutions.

### Strengths
1. The paper introduces a new problem, Discrete Representation Inconsistency (DRI), in discrete audio tokens for neural codec language models, proposing two effective methods, Slice-Consistency and Perturbation-Consistency, to improve token consistency while maintaining audio quality.

2. Thorough experiments demonstrate that the proposed methods effectively reduce DRI and improve key metrics like Word Error Rate (WER) and speaker similarity, validated on both small (LibriTTS) and large (MLS) datasets.

3. The paper is well-organized and easy to follow, clearly explaining the DRI problem, proposed solutions, and their effectiveness, making the study accessible and understandable.

### Weaknesses
1. Lack of Direct Evidence Linking Codec Inconsistency to Performance：The paper says that codec inconsistency causes omissions and repetitions in outputs, but there’s no direct evidence showing how much this affects performance. While the DRI issue is clear, its exact impact on results isn’t. Adding experiments that compare models trained with consistent and inconsistent audio tokens would make this point stronger.

2. Alternative Perturbations and the Importance of Perturbation-Consistency：The paper doesn’t talk about whether other types of perturbations, besides spectral perturbations, could help improve consistency. It also doesn’t clearly explain why Perturbation-Consistency is necessary or how it works. Trying out other types of perturbations and explaining why spectral perturbations were chosen would make the method more convincing.

### Questions
1. The paper assumes that inconsistencies between sliced and full audio representations in codecs lead directly to issues like omissions and repetitions in LLM outputs. However, there is a lack of detailed evidence or analysis showing the extent of this impact. While codec inconsistencies are highlighted, it’s unclear how strongly this correlates with LLM performance issues. Text tokenizers, which don’t face consistency problems, still experience omissions and repetitions, suggesting that other factors could be influencing these issues. More evidence linking codec inconsistency to LLM performance would strengthen this argument.

2. The results in Table 3 show that the scenario with no slicing but with phase perturbation has the lowest consistency (around 7%), yet achieves the second-best WER. This outcome contradicts the paper’s central argument that higher consistency improves model performance. This suggests that low consistency doesn’t necessarily result in poor performance, which weakens the foundation of the paper’s claims.

3. The performance difference between SpeechTokenizer and EnCodec in Table 2 is unusually large, diverging significantly from the results in the original SpeechTokenizer paper[1]. This raises concerns about the fairness and consistency of the comparisons. Without clear justification for these differences, the validity of the conclusions drawn from these comparisons is questionable. The setup for these evaluations needs more transparency to ensure fair and reliable comparisons.
[1] Zhang X, Zhang D, Li S, et al. "Speechtokenizer: Unified speech tokenizer for speech large language models," *arXiv preprint*, arXiv:2308.16692, 2023.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper investigates the phenomenon of Discrete Representation Inconsistency (DRI) in audio token sequences generated by neural audio codecs. Unlike text tokens, which are deterministic, discrete audio tokens can vary significantly even if the perceptual audio remains identical. This inconsistency complicates the prediction of subsequent tokens in neural codec language models, leading to potential errors in audio generation. The authors propose two novel methods to address DRI: the slice-consistency and perturbation-consistency methods. They demonstrate the effectiveness of these methods through extensive experiments, showing significant improvements in token consistency, reduced Word Error Rate (WER), and enhanced speaker similarity in speech synthesis tasks.

### Strengths
1. The proposed methods are easy to understand and straightforward to implement, which enhances their practical applicability.

2. The identification and in-depth analysis of Discrete Representation Inconsistency in neural audio codecs are novel contributions that highlight an important challenge in the field.

3. The methods substantially enhance the consistency of audio tokens, leading to better performance in speech synthesis tasks, as evidenced by significant improvements in WER and speaker similarity metrics.

### Weaknesses
1, Missing Baseline Comparison

In your discussion of WER improvements, your paper lacks a comprehensive comparison with other works that have also improved VALL-E to reduce WER. Since many follow-up studies on VALL-E aim to lower WER, it’s important to systematically compare your method with these works to highlight your approach's advantages. 

For example, MELLE [1] uses mel-spectrogram to replace codec enhancing model robustness. VALL-E 2 [2] proposes the repetition aware sampling and grouped code modeling to enhance the stability. 


2, Evaluation Metric

Your paper uses different WER and SIM metrics compared to those used in the original VALL-E (WavLM-TDNN,  HuBERT-Large-finetune) and its subsequent improvement papers. This inconsistency makes it difficult for readers to understand the extent of your improvements, leading to confusion. It’s recommended to align your metrics with those in the original VALL-E paper for clearer comparative analysis.

3, Fairness of Baseline Experiments


There are concerns about the fairness of your baseline experiments. It is unclear whether the training configurations for your VALL-E model using your codec were the same as those for the baseline VALL-E models, such as the one using SpeechTokenizer. Ensuring consistent training settings is crucial for a fair comparison.

Additionally, it is puzzling that the WER of your baseline codec outperforms that of SpeechTokenizer. From my understanding, VALL-E with SpeechTokenizer typically achieves a lower WER compared to standard codecs.


[1] Meng L, Zhou L, Liu S, et al. Autoregressive speech synthesis without vector quantization[J]. arXiv preprint arXiv:2407.08551, 2024.


[2] Chen S, Liu S, Zhou L, et al. VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers[J]. arXiv preprint arXiv:2406.05370, 2024.

### Questions
Consistency Accuracy and WER Correlation

Is consistency accuracy truly correlated with WER? In Figure 4, SpeechTokenizer performs worse than EnCodec and DAC in terms of consistency accuracy, which seems counterintuitive. From my experience, when training VALL-E, SpeechTokenizer typically yields better WER results compared to EnCodec and DAC.

### Soundness
3

### Presentation
3

### Contribution
2
