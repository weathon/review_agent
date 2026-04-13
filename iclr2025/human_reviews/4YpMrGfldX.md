## Human Reviewer 1

### Summary
The authors present an encoder-decoder transformer-based architecture for encoding speech at very low bitrate, below 1kbps. The Transformer Audio AutoEncoder (TAAE) uses a polyphase time-frequency representation to perform downsampling/upsampling before the enocder and after the decoder of TAAE. Finite Scalar Quantization (FSQ) is employed within the encoder/decoder bottleneck to mitigate codebook underutilization typically seen with vector quantized (VQ) and residual vector quantized (RVQ) approaches. The authors combine an L1 discriminator feature loss with decaying L1 waveform loss and late perceptual reconstruction loss for training. The TAAE is trained on 105k hours of English speech sampled at 16kHz. The reconstruction capability of TAAE is compared to the Descript Audio Codec (DAC), Encodec, SpeechTokenizer, SematiCodec, and Mimi. A mean opinion score (MOS) is also produced from a perceptual evaluation comprised of 23 participants comparing TAAE to Mimi and SemantiCodec. The authors demonstrate that TAAE obtains better reconstruction performance according to both objective measures and MOS. The authors also demonstrate that one variant of the TAAE codebook attains 98% utilization.

### Strengths
Originality:
The results presented in Appendix B are enlightening regarding the use of non-power-scaled FFT sizes. The FSQ-based bottleneck also seems to overcome common issues in the training of RVQ systems. 

Quality:
The authors provide a wide variety of objective assessments for their architecture's performance. The authors also do a good job of citing current literature.

Clarity: 
The authors very clearly describe their architecture and the motivations for their architectural decisions. The appendices were well organized and helpful.

Significance:
Appendix B and the FSQ bottleneck are worthwhile contributions.

### Weaknesses
I do not think it is novel to scale the parameter count of a neural network autoencoder and demonstrate better compression ratios/reconstruction compared to smaller architectures.  This is a well known result.

I do not think it is novel to restrict the domain of a neural network autoencoder in comparison to another architecture trained on a more general domain and demonstrate better compression ratios/reconstruction. This is a well known result.

By restricting the domain of their speech audio corpus to English speech, the authors have produced an English speech audio codec. In order to claim that this is a "speech codec," the authors should evaluate on non-English speech to demonstrate generalization capabilities.  

I do not think DAC, Encodec, and SemantiCodec are reasonable baselines to compare to, as none claim to be English speech codecs. 

Mimi focuses on streaming and causality with 1/10 the parameter count of TAAE, which makes no claims regarding streaming capability. This also leads to an odd comparison as the goals of Mimi and TAAE are not aligned.  

It is unclear why SpechTokenizer was left out of the perceptual evaluation, as it is the most comparable to TAAE in terms of architecture and training domain. Comparison to SpeechTokenizer could also boost claims that the FSQ significantly outperforms RVQ schema.  

I think the presented MOS scores are confusing, as MOSNet has estimated the MOS closer to 3 than to 5. A MUSHRA evaluation should have been used instead for pairwise comparison between codecs and is standard in the literature cited in this paper. Furthermore, the authors should include demographic breakdowns of the perceptual evaluation, as well as a description of the listening setup, as is standard for speech codecs.

### Questions
Why was SpeechTokenizer not included in the perceptual evaluation?

How do the design goals of Mimi align with that of TAAE? Why is Mimi a good baseline comparison to TAAE?

Why was a MOS evaluation chosen instead of MUSHRA?

How do you explain the gap in your perceptual evaluation MOS score and the estimate provided by MOSNet? 

Who was included in the perceptual evaluation, and what was their listening setup?

How does TAAE perform on non-English speech? And how does that compare to the more generalist NAC?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposed  TAAE, an audio codec model that uses Transformer encoders as the main building block to replace conventional convolution-based modules. To accommodate the choice, TAAE performs downsampling mainly by patchifying the time domain signal and training transformer encoder stacks on top of the downsampled sequence. For discretizing audio, TAAE relied on FSQ-based bottleneck that approximates continuous low-dimensional latent numerically. Experiment results show TAAE achieved outstanding speech quality on autoencoding at a significantly lower bit rate comparing to existing models.

### Strengths
- The idea of using Transformer and the main architecture for the neural audio codec learning is novel and well executed.
- Judging from the audio samples on the demo page and MOS study, TAAE is clearly state-of-the-art in low bit rate speech compression.
- This paper provided a lot of detailed knowledge, empirical findings, and engineering improvements that can truly benefit the audio codec research community. I personally learned a lot in the details such as the discussion on systematic bias of discriminator, choice of filterbank, observation on the latent representation of silence frames with self-attention, etc.

---

### Justification for rating
Overall, I believe this work is novel enough and provides solid contributions to the field.
However, some improvement might be necessary (see weaknesses below).
If the authors can properly address these concerns and update the submission accordingly, I would be more than happy to raise my rating on this paper.

### Weaknesses
Given the main contribution of this work is in exploring an alternative architecture for codec models, completeness in terms of design details and reproducibility are expected. In contrast, I found a lot of details missing or vague. (Although the authors state the code will be released later, the paper itself should still be comprehensive alone.) Here are some examples:

---

> ($\S$2.1) ... Instead we raise the $\epsilon$ constant used in the calculation of normalization factors in the layer norm blocks ... allows convergence of the architecture.

This appears to be an interesting observation and a critical hyper-parameter for training the model as the authors spent a paragraph discussing it, but neither the exact value nor the study/experiment on $\epsilon$  is provided.

---

> ($\S$2.4)...For training the codec itself, we primarily use a normalized feature-matching L1 loss on the per-layer features of the discriminator network ... In addition we found it beneficial to include a traditional L1 reconstruction loss to boost convergence at the beginning of the training process ...

The overall objective of the model is not explicitly given but described in a more hand-wavy style instead, which could easily lead to misunderstanding. The full objective should be listed explicitly together with the weight/importance for each term/component in the main paper or appendix.

---

> ($\S$2.1) ... The self-attention uses a sliding window of size 128, to restrict receptive field and aid generalization of the architecture to arbitrary length sequences. 

This choice appears as one simple sentence, but self-attention is the key difference between TAAE and prior works, which changes the properties of the model dramatically.
If my understanding is correct, this means the receptive field of the first layer is already 2.56 seconds (128 frames $\times$ 20 ms-per-frame), and the number doubles for every layer. It is obvious that TAAE has a much larger receptive field size comparing to convolution-based models. While this is an advantage, it could also lead to some problems that are not discussed in the paper.
 
- What is the trade-off between length generalization and sliding window size for TAAE? How do time complexity and empirical inference time change accordingly? How do these numbers compare to those of CNN-based models?
- Beyond length generalization, can TAAE perform streaming encoding/decoding (as most of the existing works compared in this paper)?
  - If so, what is the size receptive field? how does it affect the latency of the codec? how does it compares to conventional CNN-based codec models?
  - If not, this should still be explicitly discussed as a limitation of the proposed framework in the paper.

These are just some examples. In short, I believe the fundamental differences between TAAE and CNN-based codec models should be discussed in the paper more throughout and carefully. Both advantages and disadvantages should be clearly stated and summarized in the main body of the paper.

---

I believe these concerns can all be addressed without additional training, thus should be easy enough to complete within the rebuttal period.

### Questions
(please see weaknesses above)

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 3

### Summary
This work describes an approach to leverage scaled transformer model architecture to achieve low-bitrate and high-quality speech coding. Different from conventional neural audio codec approaches, this work leverages the finite scalar quantization (FSQ) based bottleneck to achieve high codebook utilization rate and avoid the difficulty in training a VQ-VAE auto-encoder. Experimental results show this works outperform existing baselines in both objective and subject tests.

### Strengths
This paper is well written and very clear to follow. In the introduction part, it clearly presents the motivations and has an excellent survey of the existing methods. 

Though using transformers to scale and leverage FSQ for high codebook utilization is not something new, this paper presents the motivations of these changes, the associated challenges and their mitigations. This paper also introduces a new method so that FSQ can be used in a similar way as RVQ where a varying bits-per-second rate can be achieved. 

This paper presents strong experiment results, significant improving over the existing baselines (but at the cost of increased computation and latency).

### Weaknesses
If I understand the proposed model correctly, it is based on transformer layer with a local attention of 128 (both left and right), which means different from DAC/Encodec/Mimi etc which use causal encoders, the encoder in the proposed method is not causal, and it will introduce a latency up to the patch length (which is 320/16k ~ 20ms?). It would be great if the author can present the results with causal encoder so that it can be compared with DAC/Encodec/Mimi in a relative fair comparison (apart from the model size difference).

### Questions
N/A

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper presents a new Transformer architecture for speech coding. It is characterized by the new scalar quantization method performed in a dimension-reduced space, which showed improved coding gain compared to other methods that are based on residual vector quantization. The paper also provides various training strategies that appear to be useful for neural codec training.

### Strengths
- The proposed system appears to work well according to the objective metrics and subjective tests.
- The proposed FSQ idea seems to be a solid quantization option, improving the codebook utilization.
- The authors put a lot of effort in making it more scalable by adding multiple levels of quantization.

### Weaknesses
- The proposed method relies on the dimension reduction part for its dimension-specific scalar quantization to work. And that's why they could achieve higher codebook utilization. Meanwhile, there is also a trend that higher codebook utilization leads to lower coding gain if entropy coding is applied after tokenization. Indeed, the paper does not mention anything about Huffman coding results, which the proposed method might not be able to take advantage of due to the low dimensionality and high codebook utilization. At the same time, the RVQ-based ones might have a better chance of compressing more via Huffman coding. I wish the paper provided an in-depth discussion about it. In my opinion, all the coding gain and performance-related arguments must be based on the entropy-coded bitrates of all codecs mentioned. 

- The other main criticism is that the proposed model is just a lot bigger than the other models. I don't mean to argue that a bigger codec necessarily results in a better coding gain, but in general, it is also true that there is a relation. I wish the paper had provided an ablation test that investigated the impact of the different sizes of the proposed model. 

- The paper provides various tips and useful information about their model training, but they are scattered in different places without a clear organization.

### Questions
- How does it compare to the HuBERT-based codec? HuBERT can be as large as the proposed model (its X-Large version), and can turn into a codec with a vocoder attached to it as shown in [a].

[a] A. Polyak et al. “Speech Resynthesis from Discrete Disentangled Self-Supervised Representations,” Interspeech 2021

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4