# DiFlow-TTS: Compact and Low-Latency Zero-Shot Text-to-Speech with Factorized Discrete Flow Matching

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 2

## Abstract
Despite flow matching and diffusion models having emerged as powerful generative paradigms that advance zero-shot text-to-speech (TTS) systems in continuous settings, they continue to fall short in capturing high-quality speech attributes such as naturalness, similarity, and prosody. A key reason for this limitation is that continuous representations often entangle these attributes, making fine-grained control and generation more difficult. Discrete codec representations offer a promising alternative, yet most flow-based methods embed tokens into a continuous space before applying flow matching, diminishing the benefits of discrete data. In this work, we present DiFlow-TTS, which, to the best of our knowledge, is the first model to investigate discrete flow matching directly to generate high-quality speech from discrete inputs. Leveraging factorized speech attributes, DiFlow-TTS introduces a factorized flow prediction mechanism that simultaneously predicts prosody and acoustic detail through separate heads, enabling explicit modeling of aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS delivers strong performance across several metrics, while maintaining a compact model size up to 11.7 times smaller and low-latency inference that generates speech up to 34 times faster than recent state-of-the-art baselines. Code and audio samples are available on our demo page: https://diflow-tts.github.io

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces DiFlow-TTS, a zero-shot TTS system applying discrete flow matching on FACodec tokens. In specific, the paper leverages a Phoneme-Content Mapper for accurate semantic alignment and a Factorized Discrete Flow Denoiser that uses separate prediction heads to explicitly model distinct speech aspects. Experimental results demonstrate that DiFlow-TTS achieves state-of-the-art performance in naturalness, content accuracy, and prosody preservation while maintaining a significantly smaller model size (up to 11.7x smaller) and achieving substantially lower inference latency (up to 34x faster) than competing baselines.

### Strengths
1.The paper is the first to use discrete flow matching for zero-shot TTS

2.The paper achieves enhanced quality with smaller model size and faster inference speed.

3.The code is released.

### Weaknesses
1.The paper's central novelty lies in applying a discrete generative model to factorized tokens. However, this general approach has been explored using discrete diffusion models, most notably in NaturalSpeech 3 which introduced the FACodec. To better isolate the contribution of using discrete flow matching(DFM), the paper would be significantly strengthened by: (1) explicitly discussing the theoretical and practical differences between DFM and discrete diffusion in the methodology section, and (2) including an experimental baseline using a discrete diffusion model trained on the same FACodec tokens to provide a direct comparison of the two generative frameworks.

2.A notable weakness is the model's performance on speaker similarity, as measured by SIM-O, which lags considerably behind state-of-the-art models like MaskGCT. Given that high-fidelity voice cloning is a central goal of zero-shot TTS, this performance gap is significant. 

3.There is an inconsistency in the paper's framing. The authors criticize prior works for the "redundant" step of embedding discrete inputs to a continuous space (L55-57), claiming their "direct discrete formulation removes this conversion." However, their own method employs "prosody and acoustic embedders" (L236), which performs an analogous function. The manuscript's claims should be revised to reflect this more precise and accurate distinction.

4.The assertion that continuous representations are inherently more prone to attribute entanglement than discrete ones (Abstract and L51) is a strong claim that is not empirically substantiated within the paper. To validate this claim, a comparative experiment modeling the continuous embeddings from the VQ-VAE codebooks in FACodec would be necessary. Otherwise, the authors should moderate this claim.

### Questions
See weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents DiFlow-TTS, a non-autoregressive, zero-shot text-to-speech system that introduces a novel approach by leveraging purely Discrete Flow Matching (DFM). The core idea is to model the generation process directly in the discrete space of factorized speech tokens, avoiding the common practice of mapping them to a continuous space. The architecture is composed of a speech tokenizer that disentangles audio into prosody, content, and acoustic attributes; a Phoneme-Content Mapper for robust text alignment; and the central Factorized Discrete Flow Denoiser (FDFD). The FDFD uses a Diffusion Transformer (DiT) architecture with dedicated prediction heads to explicitly model the distributions of prosody and acoustic details. The authors provide a thorough empirical evaluation demonstrating that DiFlow-TTS achieves state-of-the-art performance while being substantially smaller and faster than a range of baseline models.

### Strengths
1. Novel and Well-Motivated Application of DFM: The primary strength of this work lies in its innovative application of a purely discrete generative framework to speech synthesis. By demonstrating the viability of DFM directly on factorized speech tokens, the paper presents a principled and elegant alternative to continuous diffusion models, simplifying the generation pipeline and reducing computational overhead. This is a timely and significant contribution to the field.

2. Excellent Performance and Efficiency Balance: The empirical results are very strong. The paper shows that DiFlow-TTS achieves top-tier performance when compared against the chosen baselines. This high quality is achieved with remarkable efficiency. The model offers much lower latency than competitors, establishing a very favorable quality-efficiency trade-off.

3. Rigorous Experimental Validation: The paper is supported by a comprehensive and well-designed set of experiments. The authors benchmark against a diverse suite of models and employ a wide range of metrics, including objective, subjective, and prosody-specific evaluations. The ablation studies are particularly effective, systematically confirming the importance of the key design choices, such as the factorized prediction heads and the inclusion of various conditioning signals.

### Weaknesses
1. Limited Competitiveness of Baselines: The primary weakness of this paper is that the selected set of baselines may not represent the current state-of-the-art in text-to-speech synthesis. While models like VALL-E and NaturalSpeech 2 were foundational, the field has seen rapid progress with the advent of powerful, large-language-model-based systems. The omission of more recent and stronger competitors makes it difficult to accurately situate DiFlow-TTS's performance. This weakens the central claim of superiority, as the reported performance gap might be significantly smaller when compared against true state-of-the-art systems.

2. Discrepancy in Speaker Similarity Metrics: While the model achieves a top score in the subjective evaluation of speaker similarity (MOS), the objective SIM-O metric remains noticeably lower than that of several key baselines (e.g., F5-TTS, MaskGCT). 

3. Reliance on Pre-Existing Representations Constrains Novelty: The paper's core innovation lies in applying DFM to a pre-existing, factorized token representation from FACodec. While this is a valid and effective approach, it frames the work's novelty as being primarily in the generative methodology rather than in the end-to-end synthesis pipeline or the speech representation itself. The impressive performance is therefore a joint product of the proposed DFM model and the powerful capabilities of the external codec, which can make it difficult to isolate the precise contribution of the flow matching component to the final quality.

### Questions
1. The comparison with OZSpeech is crucial for evaluating the paper's claims on efficiency. Table 4 reveals that DiFlow-TTS achieves a UTMOS of 2.904 with a single function evaluation (NFE=1). In contrast, Table 3 shows that OZSpeech, a continuous flow-matching model, reaches a higher UTMOS of 3.15 in the same single-step setting. This suggests that at the lowest possible latency (NFE=1), the continuous flow matching approach currently outperforms the proposed discrete method in terms of speech naturalness.

To make a complete assessment, could the authors provide a full evaluation of DiFlow-TTS at NFE=1, including all key metrics from Table 1 (WER, SIM-O, F0/Energy RMSE)? Such a comparison is critical for justifying the central motivation of the paper. If the discrete flow matching approach does not demonstrate a clear advantage at its most efficient, single-step inference point, what is the compelling reason to prefer it over established continuous flow matching methods for low-latency TTS?

2. The field of generative speech is advancing very rapidly. The current baseline set is strong, but have you considered benchmarking against more recent and powerful LLM-based systems like CosyVoice 2 [1] or Spark-TTS [2]? Including them, even if trained on larger datasets, would provide a more complete picture of the current state of the art.

3. The ablation study effectively demonstrates the importance of attribute-type embeddings. Would it be possible to conduct a finer-grained study by removing only the prosody attribute embedding or only the acoustic attribute embedding? This could provide deeper insights into how each attribute-specific signal contributes to the final synthesis quality.

[1] Du, Zhihao, et al. "Cosyvoice 2: Scalable streaming speech synthesis with large language models." arXiv preprint arXiv:2412.10117 (2024). 
[2] Wang, Xinsheng, et al. "Spark-tts: An efficient llm-based text-to-speech model with single-stream decoupled speech tokens." arXiv preprint arXiv:2503.01710 (2025).

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces DiFlow-TTS, a zero-shot text-to-speech framework that performs flow matching directly in the discrete space of factorized codec tokens, avoiding the continuous embeddings used in prior “discrete” flow-based methods. The model leverages a Phoneme-Content Mapper (PCM) to align phoneme sequences with discrete speech tokens for semantic grounding, and a Factorized Discrete Flow Denoiser (FDFD) with separate prediction heads for prosody and acoustic details, enabling explicit modeling of aspect-specific distributions. Experimental results show that DiFlow-TTS achieves strong speech quality, naturalness, and prosody preservation, while maintaining a compact model (up to 11.7× smaller) and low-latency inference (up to 34× faster) than state-of-the-art baselines. The approach provides an efficient and controllable discrete generative framework for zero-shot speech synthesis.

### Strengths
S1:The paper explores discrete flow matching (DFM) for zero-shot TTS — an area still dominated by diffusion- or AR-based models. Applying DFM directly on factorized codec tokens (prosody, content, acoustic) without continuous embeddings is novel in this context. The proposed Factorized Discrete Flow Denoiser (FDFD) and Phoneme-Content Mapper (PCM) together form a coherent pipeline that cleanly disentangles speech attributes.

S2:The method is generally well-formulated, and the paper carefully adapts the discrete flow formulation (Gat et al., 2024) to token-based speech synthesis. The derivations are mathematically consistent, and most architectural details are explicit. Ablation studies are informative and show that each component contributes meaningfully (Table 5). DiFlow-TTS shows competitive or superior naturalness (UTMOS/MOS) compared to strong baselines like F5-TTS and MaskGCT, with a compact 164M parameter size.

### Weaknesses
W1: The definition of the probability velocity in Eq. (1) appears to directly subtract a Kronecker delta term, rather than the marginal path distribution $p_t(x^i \mid x_t)$. While this form is mathematically consistent with Gat et al. (2024, Eq. 24) under the deterministic convex-interpolant path assumption (i.e., $p_t(x^i \mid x_t) = \delta_{x_t}(x^i)$), it would improve clarity if the authors explicitly stated this assumption. Otherwise, readers might interpret the formula as omitting $p_t$ from the more general flow-matching definition.

W2: The proposed Phoneme-Content Mapper (PCM) appears to follow the same structure as conventional duration-based alignment modules used in models such as FastSpeech2, consisting of a phoneme encoder, duration predictor, length regulator, and feed-forward Transformer layers. While it serves an important role in aligning textual phonemes with discrete codec tokens (which differ from mel frames), the architecture itself is not novel. I suggest clarifying that PCM is an adaptation of existing duration-alignment mechanisms to discrete token modeling, rather than a core methodological contribution.

W3:The formulation in Eq. (3) is consistent in terms of dimensionality and model structure.
The projection branch $\mathcal{H}\rho(\cdot)$ outputs content embeddings
$\mathbf{h}c \in \mathbb{R}^{n\times L\times D}$,
and the content head $\mathcal{G}\varphi(\cdot)$ produces logits over a vocabulary of size $v$,
$\mathcal{G}\varphi(\mathbf{h}) \in \mathbb{R}^{n\times L\times v}$.
However, the notation
$p(x^c \mid \mathbf{h}; \varphi) = \mathcal{G}\varphi(\mathbf{h})$
is slightly imprecise: since $\mathcal{G}\varphi(\cdot)$ outputs unnormalized logits rather than a normalized probability distribution,
a more accurate form would be
$p(x^c \mid \mathbf{h}; \varphi) = \text{softmax}(\mathcal{G}_\varphi(\mathbf{h}))$.

W4:The equation in section 3.4 is mathematically consistent, but the accompanying text is misleading: the embeddings are concatenated along the quantizer (attribute) dimension, not the temporal one. The shape $(m+n+k)\times(L_p+L)\times D$ confirms this.

W5:WER is often reported as a fraction (e.g., 0.05 = 5%), but this is not stated in the table. Please clarify the unit, as readers may misinterpret the numbers otherwise.

W6: The comparison with OZSpeech (145 M params, NFE = 1, RTF = 0.03, WER = 0.05) suggests that DiFlow-TTS does not exhibit a clear advantage in speed or word-error performance at a similar model scale. While the UTMOS score is slightly higher, the difference appears modest given the comparable inference cost. I encourage the authors to clarify what specific architectural or training differences contribute to the claimed efficiency gains.

### Questions
Q1:Could the authors include comparisons against other recent zero-shot TTS models (e.g., NaturalSpeech 3 (Ju et al., 2024), E2 TTS (Eskimez et al., 2024), ZipVoice (Zhu et al., 2025), VALL-E 2 (Chen et al., 2024)) in terms of model size, inference steps, RTF and WER? This would strengthen the claim that the proposed method offers a distinct speed-quality trade-off.
Q2: The reported WER values for several baselines (e.g., VALL-E and NaturalSpeech) differ by nearly an order of magnitude from those reported in other papers such as NaturalSpeech 3 (e.g., VALL-E ≈ 6 % vs. 0.19 here). Even if we ignore unit scaling, the absolute values seem substantially lower. Could the authors clarify which ASR system was used to compute WER and whether their evaluation follows the same protocol as previous works. Using a consistent WER computation method would make cross-paper comparisons more meaningful.
Q3: The paper’s core motivation is that discrete, factorized representations enable better attribute disentanglement (prosody, content, and acoustic detail) compared to continuous embeddings. While Table 5 and the Effect of Each Component analysis show that removing individual heads degrades perceptual metrics, these results only confirm that each branch contributes to quality rather than directly validating disentanglement. Additional evidence or analyses would be helpful to substantiate this key motivation.

### Soundness
3

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
3

### Summary
This paper proposes DiFlow-TTS, a zero-shot TTS system that performs purely discrete flow matching (DFM) over factorized speech tokens (content, prosody, acoustic). A Phoneme–Content Mapper (PCM) maps phonemes to discrete content tokens and embeddings (with duration/length regulation). A Factorized Discrete Flow Denoiser (FDFD) then predicts prosody and acoustic tokens via separate probability-velocity heads, conditioned on the content embeddings, a speaker embedding, and reference-prompt tokens. On LibriTTS training and LibriSpeech test-clean evaluation with 3-second prompts, the paper reports: UTMOS 3.98, WER 0.05, SIM-O 0.45 / SIM-R 0.54, F0 Acc 0.88 / RMSE 7.97, Energy Acc 0.73 / RMSE 0.007; and RTF 0.066 with 164 M params at NFE=16 (second-fastest behind a 1-step baseline). Ablations show large drops without speaker or content embeddings; NFE sweeps show strong gains up to ~8–16 steps with diminishing returns thereafter.

### Strengths
1. First reported discrete flow-matching approach to TTS with explicit factorization and separate prosody/acoustic heads.
 2. compact and fast relative to many contemporaries (RTF~0.066@16 NFE and ~164 M params); speed-quality trade-off is explored via NFE.

### Weaknesses
1. RTF comparisons are informative but would be stronger with fully standardized hardware conditions reported in-paper.
 2. The paper does not include controlled comparisons with strong discrete-token systems such as MaskGCT and IndexTTS2.
 3. The audio samples on the demo page sound relatively low in quality.

### Questions
1. I am familiar with several reported baselines. In Table 3, you list F5-TTS with WER = 0.24 (≈24%) at 32 NFEs, whereas the F5-TTS paper reports WER ≈ 2.42% (32 NFEs) and 2.53% (16 NFEs, RTF ≈ 0.15) on LibriSpeech-PC test-clean. These are an order of magnitude apart. Please precisely document your training/inference pipeline for each baseline (checkpoints used, NFE, solver, Sway Sampling on/off and schedule, CFG, ASR model/version, text normalization, prompt trimming/segmentation, random seeds, hardware), and provide a reproducible experiment sheet (per-model scripts/configs) so the community can replicate the exact numbers in your tables.
2. Factorized heads are reasonable, but the paper evaluates disentanglement indirectly. It lacks causal or information-theoretic evidence to substantiate controllability and orthogonality of factors.

### Soundness
2

### Presentation
2

### Contribution
2
