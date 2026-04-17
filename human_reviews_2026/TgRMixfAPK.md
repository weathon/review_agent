# Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Bandwidth extension, the task of reconstructing the high-frequency components of an audio signal from its low-pass counterpart, is a long-standing problem in audio processing. While traditional approaches have evolved alongside the broader trends in signal processing, recent advances in neural architectures have significantly improved performance across a wide range of audio tasks. In this work, we extend these advances by framing bandwidth extension as an audio token prediction problem. Specifically, we train a transformer-based language model on the discrete representations produced by a disentangled neural audio codec, where the disentanglement is guided by a Harmonic–Percussive decomposition of the input signals, highlighting spectral structures particularly relevant for bandwidth extension. Our approach introduces a novel codec design that explicitly accounts for the downstream token prediction task, enabling a more effective coupling between codec structure and transformer modeling. This joint design yields high-quality reconstructions of the original signal, as measured by both objective metrics and subjective evaluations. These results highlight the importance of aligning codec disentanglement and representation learning with the generative modeling stage, and demonstrate the potential of global, representation-aware design for advancing bandwidth extension.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HP-codecX, a novel two-stage approach for audio bandwidth extension. The method first employs a neural audio codec, HP-codec, which is designed to produce a disentangled latent representation. This disentanglement is structured along two axes: separating low and high frequency bands, and further decomposing each band's representation into harmonic, percussive, and residual components. In the second stage, a multi-branch Transformer-based language model is trained on these discrete tokens to predict the high-frequency tokens from the low-frequency ones. The authors present extensive experiments, including objective metrics and subjective MUSHRA listening tests, demonstrating that their method achieves state-of-the-art performance compared to recent strong baselines like Apollo and AudioSR.

### Strengths
The idea of structuring the latent space of a neural audio codec to explicitly align with a downstream task is a very strong and promising direction. The authors' design, which introduces a harmonic-percussive decomposition as an inductive bias, is well-motivated by principles of audio signal processing and aims to create a more predictable and interpretable representation for the language model. The empirical results presented are a clear strength of this work. The proposed HP-codecX consistently outperforms strong, recent baselines on both in-domain and out-of-domain datasets, as evidenced by a comprehensive suite of objective metrics and, importantly, a well-conducted MUSHRA listening test. The thoroughness of the evaluation adds significant credibility to the paper's claims of state-of-the-art performance.

### Weaknesses
While the results are impressive, I find the paper's fundamental premise to be insufficiently justified, which constitutes a significant weakness. The entire approach is built on using a neural audio codec, which is an inherently lossy compression process. The paper does not adequately explain why introducing this information bottleneck is a desirable step for a high-fidelity restoration task like bandwidth extension. It seems counter-intuitive to first discard information through compression, only to then try and hallucinate even more information (the high frequencies). The advantage of this token-based prediction over more direct spectrogram- or waveform-based inpainting/generation methods is not made clear.
This motivational weakness is further exacerbated by a critical lack of detail regarding the cornerstone of the method: the harmonic-percussive-residual decomposition. The paper states that the training is guided by this decomposition, but it never specifies how this decomposition is performed. Is it a classic algorithm like median filtering? Is it a learned component? How are the decomposed signals used to train the respective "harmonic," "percussive," and "residual" RVQ streams? Without these crucial details, the central claim of achieving "semantically informed disentanglement" is unsubstantiated and the method is not fully reproducible.

### Questions
To better understand the contributions and rationale of your work, I would appreciate clarification on the following points:
1. Could you please elaborate on the fundamental advantage of using a lossy codec as a pre-processor for bandwidth extension? Why is a compressed, discrete token representation a better starting point for this task than the original, continuous low-band spectrogram or waveform, which contains more pristine information?
2. For the training of the HP-codec, you mention specific iterations for harmonic and percussive components. Could you provide the specific details of the harmonic-percussive-residual decomposition algorithm used to generate the training data for these steps? This information is essential for understanding your method and for reproducibility.
3. Following up on the above, the ablation in Table 5 is interesting. It shows that the harmonic and percussive streams are indeed better at reconstructing their respective signal types. However, given the unclear decomposition process, could this result simply be a product of the training data curation? More importantly, what happens to the overall bandwidth extension quality if you ablate one of the streams (e.g., predict only with harmonic and residual tokens)? This would help quantify the actual contribution of each "semantic" branch to the downstream task.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes HP-codecX, a bandwidth-extension framework built on a two-branch neural codec (at 16 kHz and 48 kHz) that internally disentangles tokens into harmonic, percussive, and residual sections. A lightweight token-level language model (implemented as three small transformer modules) predicts the missing 48 kHz tokens conditioned on the 16 kHz tokens. The 48 kHz decoder then reconstructs the wideband signal. The authors argue that the semantic disentangling improves modeling difficulty by routing different spectral/temporal structures into separate RVQ pathways. They show objective and subjective improvements over AudioSR and Apollo, provide qualitative evidence of specialization, and present a small MUSHRA-style listening test.

### Strengths
Conceptually aligned inductive bias. Splitting harmonic/percussive/residual structure matches known spectral decompositions in music/audio and is a reasonable prior for bandwidth extension.

* The model decomposition (two-branch codec -> semantic RVQ -> LM) is straightforward and can be done with standard components.

* Ablations on harmonic and percussive inputs show that the intended sections contribute as designed.

* The proposed method outperforms strong baselines on many objective metrics and is preferred in a small subjective test.

* Alternating specialization updates and staged training is practical and computationally moderate.

### Weaknesses
* The system is essentially a composition of existing components (RVQ codec + token LM).
The semantic split is a hand-crafted heuristic, not a new modeling paradigm, and the architecture is also taken from existing work. Novelty overall is therefore limited.

* One of the core claims regarding the necessity of the semantic split is not demonstrated. The ablation removes the semantic split but replaces three deeper transformers with a single-layer LM, and only makes very vague claims as to why ("Only the single-layer transformer was able to learn non-trivial representations"). As this is one of the core components that the paper claims is necessary, this ablation needs to be done properly and in a setting where the capacity of the ablation vs. the baseline is comparable. It is also possible that the split into more codebooks/tokens is what yields better performance and not necessarily that the split is done semantically.

* The paper does not evaluate against mmodels such as n Grumiaux & Lagrange  and it is not clear why. The paper apparently even uses parts from this paper for the data. This makes the claim of SOTA performance a bit questionable. 

* Code is not provided (only “upon acceptance”), which makes it impossible to verify and understand all details.

### Questions
* Why was the non-semantic baseline restricted to a single transformer layer?

* Will you add a parameter-matched unified LM baseline to properly isolate semantic structure effects?

* Why were some other baselines omitted?

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
4

### Summary
This paper focuses on the music bandwidth extention (Music BWE), with sampling rate (Fs) = 16kHz as the input, and Fs = 48kHz (full band) as the targeted output.

Technically, this paper adopts the language modeling (LM) approach for audio restoration, which trains an LM with a discrete neural audio codec.

The paper compares the proposed method HP-Codec-X with two open-source BWE model, APOLLO (non-generative model) and AudioSR (a diffusion model trained inside a Mel-VAE latent space, the Mel-to-wav conversion is done by a 48kHz HiFi-GAN vocoder).

However, there are many weaknesses, including missing baseline / ablation study, too limited problem setting, among others.

As a reviewer, I tend to reject this paper.

### Strengths
- The model can do 16khz-to-48khz BWE

### Weaknesses
## Missing baseline for the codec part
In Sec.3.1.1, authors explicitly mentioned that the architecture is inspired by DAC. Given this relationship, it is essential to evaluate the performance of a standard DAC model in Table.1. Unfortunately, DAC is not included.

As a reader, I would be interested in the benchmark of more standard codecs, including DAC, EnCodec, or more recent MuCodec and SpectroStream. All these models offer pretrained 48kHz or 44.1kHz weights. 

Without the comparison with all these prior work, I cannot understand how well the proposed codec is.
## Missing baseline for the LM part
As mentioned above, there have been many open-source 48kHz audio codec.

Before proposing a new LM approach to do BWE, it is essential to train standard LMs on standard codecs such as EnCodec or DAC. Unfortunately, such experiments are not included.

What we can see from the evaluation is that, the proposed method outperforms APOLLO and AudioSR, but we cannot see if the proposed framework is necessary to achieve this performance.

Some speech restoration models trained with standard LM and standard codec:
1. Genhancer: https://www.isca-archive.org/interspeech_2024/yang24h_interspeech.html
2. MaskSR: https://arxiv.org/abs/2406.02092

Another audio language model that can do BWE without finetuning or training
- SpecMaskGIT: https://zzaudio.github.io/SpecMaskGIT/

Obviously, these LM-based speech restoration or audio generation models can be trained to perform BWE by changing the training data, and I would say these models are very standard and contain no tricky parts.

If these simple models already work for music BWE, what is the advantage of HP-Codec-X? The current paper cannot answer this key question.
## Improper problem setting
Music BWE is an important task. However, this paper only handles a fixed 16kHz input, which is too constrained compared to prior arts, and is also too far from real-world scenarios.

For example, AudioSR handles variable bandwidth input.

Considering this point, it is unfair to compare the proposed method (fixed bandwidth input) with other methods that support variable bandwidth input.

### Questions
Why standard codecs and standard language model designs were ignored in this paper?

Why only consider Fs=16kHz as  input, instead of variable input bandwidth?

### Soundness
2

### Presentation
2

### Contribution
1
