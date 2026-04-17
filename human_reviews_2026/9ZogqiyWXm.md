# Token-Based Audio Inpainting via Discrete Diffusion

- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Audio inpainting seeks to restore missing segments in degraded recordings. Previous diffusion-based methods exhibit impaired performance when the missing region is large. We introduce the first approach that applies discrete diffusion over tokenized music representations from a pre-trained audio tokenizer, enabling stable and semantically coherent restoration of long gaps. Our method further incorporates two training approaches: a derivative-based regularization loss that enforces smooth temporal dynamics, and a span-based absorbing transition that provides structured corruption during diffusion. Experiments on the MusicNet
and MAESTRO datasets with gaps up to 750ms show that our approach consistently outperforms strong baselines across range of gap lengths, for gaps of 150ms and above. This work advances musical audio restoration and introduces new directions for discrete diffusion model training. Visit our project page for examples and code.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel approach for restoring missing audio segments, using tokenized audio representations—specifically, pretrained WavTokenizer—and discrete diffusion modeling to achieve more effective inpainting of longer gaps. Training involves span-based masking as a structured corruption strategy and incorporates a derivative-regularized reconstruction loss.

Experiments conducted on two music datasets demonstrate that the method outperforms baseline approaches across three objective metrics, especially when filling gaps exceeding 200ms.

### Strengths
Well-designed system capable of handling inpainting gaps up to approximately 500ms.

### Weaknesses
The quality is heavily depends on the tokenizer or codec used.

The method lacks evaluation outside the music domain and does not consider additional conditions for music restoration.

No subjective measurements are provided, and demo samples show noticeable boundary artifacts.

### Questions
Audio inpainting can leverage diffusion models either on continuous latent spaces or discrete tokens. It would be beneficial to directly compare these two strategies—using VAE for continuous representation and a neural codec for discrete tokens—at the same frame rate.

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
This paper introduces Audio Inpainting via Discrete Diffusion (AIDD), a novel method for restoring missing segments in audio, particularly long gaps. The core contribution is being the first to apply discrete diffusion for audio inpainting, which enables more stable and semantically coherent generation compared to previous continuous-domain methods. The paper also proposes two new training techniques: 1) a span-based masking strategy for structured corruption and 2) a derivative-based regularization loss to ensure temporal smoothness. Experiments on the MusicNet and MAESTRO datasets demonstrate that AIDD outperforms strong baselines for gaps of 150 ms and longer, significantly advancing the state of the art in musical audio restoration.

### Strengths
1.The paper is the first to apply the discrete diffusion on tokenized representations for audio inpainting. 

2.The method achieves state-of-the-art results on the long-gap audio inpainting task.

3.The code will be open-sourced.

### Weaknesses
1.The evaluation lacks a subjective listening study, which is essential to validate the perceptual quality and musical plausibility of the results.

2.The paper should quantify information loss from tokenization by reporting metrics on both the original audio (as a reference ceiling) and the reconstructed audio (audio passed through the tokenizer's encoder-decoder). This would clarify the tokenizer's impact and establish the method's practical upper bound.

3.The audio sampling rates are not reported. It is unclear if the source audio is downsampled to match the WavTokenizer's 24 kHz reconstruction bandwidth, which would be a critical confounding factor affecting task difficulty.

4.There is a potential training-inference mismatch. During training, the tokenizer processes the complete audio signal before tokens are masked ("tokenize-then-mask"). At inference, it processes a signal that already contains gaps ("mask-then-tokenize"). It is unclear if the long gap introduced at inference interferes with the tokenization of other regions. This discrepancy should be discussed.

5.Key inference hyperparameters (e.g., diffusion steps, temperature, top-k) are missing, which hinders reproducibility. A latency analysis would also be beneficial to assess the method's practical usability.

### Questions
See weaknesses.

### Soundness
2

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
3

### Summary
This paper presents AIDD (Audio Inpainting via Discrete Diffusion), a model that performs audio inpainting directly in the discrete token domain instead of waveform or spectrogram space. Audio is first tokenized using a pretrained WavTokenizer, and a discrete diffusion model learns to predict masked token spans. The method introduces two main ideas: (1) Derivative-based regularization to ensure smooth temporal transitions between predicted tokens. (2) Span-based masking that masks contiguous token spans following a diffusion noise schedule, aligning the corruption process with the inpainting objective. AIDD is evaluated on MusicNet and MAESTRO, showing improved perceptual quality (FAD, ODG, LSD) on medium-to-long gaps (200–750 ms) compared to prior methods like CQT-Diff+, GACELA, and bin2bin, while being smaller and faster to train. The work contributes an efficient and conceptually clear token-level diffusion approach for long-gap audio restoration.

### Strengths
This paper presents a token-based diffusion model for audio inpainting (AIDD) that operates directly in the discrete token space rather than waveform or spectrogram domains. The idea is original in its formulation and addresses a practical limitation of prior work—difficulty maintaining long-range temporal and semantic consistency when filling large gaps. The proposed span-based masking and derivative regularization are intuitive yet effective design choices that align well with the inpainting objective. Experiments on MusicNet and MAESTRO demonstrate gains on medium- and long-gap scenarios, especially under limited computational resources. The method’s simplicity and the decision to train on single-GPU hardware make it appealing for future research and reproduction.

### Weaknesses
(1) Codec choice not sufficiently justified.
- The method relies entirely on WavTokenizer, but there are other single-codebook codecs such as UniCodec [1] that could equally serve this purpose. The paper does not explain why WavTokenizer was chosen or whether the improvements are specific to that tokenizer. A small ablation with an alternative codec would help isolate the contribution of the proposed diffusion mechanism.

(2) No human evaluation.
- The paper claims that AIDD produces perceptually natural and semantically coherent audio, yet only objective metrics (FAD, ODG, LSD) are reported. For a perceptual task like inpainting, even a small-scale human listening test would strengthen the claim substantially.

(3) Fairness and completeness of baselines.
- The paper compares AIDD against CQT-Diff+, GACELA, and bin2bin, but the comparison is not entirely fair or complete:
- Different training steps and data splits: AIDD was trained for 100 k steps on MusicNet, while CQT-Diff+ used 400 k in its original setup. On MAESTRO, AIDD trained on a private subset (not released), whereas baselines used the full dataset. This makes direct comparison difficult.
- Different modeling domains: AIDD operates in the token domain, while CQT-Diff+ and GACELA work in the spectrogram or waveform domain. Since metrics like FAD depend on the reconstructed waveform and decoder quality, comparing across such domains may not reflect pure modeling differences.

(4) Limited scope of evaluation.
The model is only tested on central silent gaps. More realistic cases—multiple gaps, noisy or partially masked regions—are not examined, leaving generalization unexplored.

[1] Jiang, Yidi, et al. "UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook." arXiv preprint arXiv:2502.20067 (2025).

### Questions
(1) Why was WavTokenizer selected over newer or higher-quality single-codebook codecs like UniCodec?

(2) Have you tested whether your derivative regularization still helps when using a different tokenizer?

(3) Could you include a small human listening study (e.g., MOS or AB preference) to verify that objective gains correlate with perceived quality?

(4) How stable are your FAD results given the relatively small sample size (~600 clips)? Any confidence intervals or bootstrap analysis?

(5) Can your span-masking strategy handle multiple or non-silent gaps, or does it assume fully silent regions only?

(6) You mention AIDD is smaller and faster than CQT-Diff+. Could you report the parameter count and inference speed to quantify this claim?

### Soundness
2

### Presentation
3

### Contribution
2
