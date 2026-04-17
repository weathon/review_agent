# Continuous Audio Language Models

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Audio Language Models (ALM) have emerged as the dominant paradigm for speech and music generation by representing audio as sequences of discrete tokens. Yet, unlike text tokens, which are invertible, audio tokens are extracted from lossy codecs with a limited bitrate. As a consequence, increasing audio quality requires generating more tokens, which imposes a trade-off between fidelity and computational cost. We address this issue by studying Continuous Audio Language Models (CALM). These models instantiate a large Transformer backbone that produces a contextual embedding at every timestep. This sequential information then conditions an MLP that generates the next continuous frame of an audio VAE through consistency modeling. By avoiding lossy compression, CALM achieves higher quality at lower computational cost than their discrete counterpart. Experiments on speech and music demonstrate improved efficiency and fidelity over state-of-the-art discrete audio language models, facilitating lightweight, high-quality audio generation.
Samples are available at iclr-continuous-audio-language-models.github.io. Finally, we release Pocket TTS, an open-source 100M-parameter text-to-speech model that can run faster than real time on a laptop CPU: github.com/kyutai-labs/pocket-tts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Discrete-token audio LMs (e.g., RVQ stacks + AR decoders) trade fidelity for latency: deeper codebooks mean more tokens and slower sampling. The paper investigates whether we can get rid of discretization and instead do autoregressive modeling in a continuous latent space while keeping quality and reducing inference cost by few-step sampling with consistency models. 
The paper contributes CALM, an autoregressive framework that operates directly on continuous VAE latents and replaces RVQ/diffusion heads with a small consistency-model MLP to enable few-step next-latent prediction. This avoids discretization while targeting real-time sampling. It introduces a dual-context design, a noise-injected long-term backbone combined with a clean short-context transformer over the last K latent, whose features are summed to serve as the condition of the head. Empirically, the consistency head reduces inference cost while maintaining or improving quality.

### Strengths
At a high level, the paper’s short-context pathway is analogous to LocDiT in DiTAR: both introduce an additional local/short-term module on top of a long-range autoregressive backbone to recover fine detail that a purely long-context model tends to miss. That said, the mechanisms and constraints differ: the proposed short-context transformer is causal and consumes a clean recent window whose features are summed with a noised long-context stream to condition a few-step consistency head. By contrast, LocDiT performs bidirectional, intra-patch diffusion with a small prefix of historical patches, and typically runs for many denoising steps. The goals align, but the causality, noise-injection, and sampling budgets (few-step consistency vs. multi-step diffusion) are importantly different.

### Weaknesses
- The paper targets continuation on speech and music, but the evaluation scope is too narrow to position the method within the broader speech modality. In practice, TTS is the most common task for audio generation models. Without a TTS evaluation, it’s difficult to compare against widely used or state-of-the-art systems. Including comparisons with strong baselines (e.g., GLM-4-Voice, DiTAR) on overlapping tasks would be required. In short, the method is well-substantiated for continuation, but its position relative to on-the-shelf speech models is unclear. Adding TTS (with direct SOTA comparisons) would make the contribution much easier to compare and significantly strengthen the paper.

- Lack of references: Suggest citing IMPACT [A] in line 53, which is also a MAR-style audio generation model.

[A] Huang, K. P., Yang, S. W., PHAN, H., Lu, B. R., Kim, B., Macha, S., ... & Wang, C. IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling. ICML 2025

### Questions
Why sample $x_t$ with $x_t = cos(t)x + sin(t) \epsilon$ instead of $x_t = (1-t) \cdot x+t \cdot \epsilon$ ?

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
To address the quality and efficiency issues of discrete tokens, the paper proposes CALM. CALM models directly in the continuous latent space and introduces a consistency model to enhance audio reconstruction quality while significantly improving inference speed. By using the Causal Short-Context Transformer module, the quality of generation is enhanced. In addition, the paper proposes practical techniques such as temperature sampling and Head Batch Multiplier to enhance training stability and controllability of generation.

### Strengths
The paper makes several improvements to existing Audio Language Models, introducing new mechanisms and training methods with a good degree of innovation. The paper is well-written, with the methods section presenting a complete mathematical process and the experimental section validating multiple improvements. CALM is of considerable importance to current audio language models.

### Weaknesses
1. Although a detailed comparison is made with the RQ-Transformer, there is insufficient comparison with other continuous audio generation models (such as SALAD, DiTAR, etc.).

2. As shown in the experimental data presented by the authors (Table 5), the Short-Context Transformer is crucial for generation quality, but the authors do not specifically explain the working mechanism of this module, nor do they explain how the absence of this module would degrade the model's generation results.

3. Similarly, the authors need to more clearly explain the specific impact mechanism of noise injection on model robustness and preferably provide some visual results.

### Questions
1. Could the authors provide a more detailed explanation of the working mechanism of the Short-Context Transformer, or validate through experiments and visual results what specific information this module captures?

2. Recently, some new continuous modeling-based audio language models have emerged, such as DiTAR and Ming-UniAudio. If possible, could the authors compare the technical similarities and differences as well as the actual performance between CALM and their models?


DiTAR: https://arxiv.org/abs/2502.03930

Ming-UniAudio: https://mdn.alipayobjects.com/cto_asrtts/uri/file/as/TR-Ming-UniAudio.pdf

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
The paper proposes CALM, a continuous autoregressive audio generation framework without discrete quantization. The idea is to first replace lossy RVQ codecs with a standard VAE, and then generate latent vectors frame-by-frame using transformers with consistency heads. Experiments show that on both speech and music continuation tasks, CALM can generate high-quality, long-context and meaningful audio contents within real-time.

### Strengths
1. Different from previous MAR-style work designed for single tasks like TTS and musical audio generation, CALM aims to build **a unified framework** for multiple audio generation tasks. The authors illustrated the effectiveness of CALM on **two challenging tasks**: speech and music continuation, demonstrating CALM’s ability to generate content with high semantic meaningfulness and acoustic quality.
2. The idea to replace sampler/diffusion/flow matching heads with consistency head is novel. This approach, along with the newly-proposed training strategy of head batch multiplier, significantly reduces inference and training cost.
3. The paper proposes an effective solution to the error accumulation problem in continuous audio generation with noise augmentation and short context transformer.

### Weaknesses
1. Although the meaningfulness and enjoyment improves upon baseline methods, the generated audio still lacks long-content semantic continuity as illustrated by samples in the demo page.
2. The proposed method is evaluated only on unconditional generation tasks, making it unclear whether the proposed method can be extended to more applicable conditioned tasks like TTS and text-to-music. This also makes it difficult to compare CALM with mainstream baselines that work on conditional generation tasks.

### Questions
1. Have the authors considered methods that directly generates mel-spectrogram, like MELLE [1] and Flow-Omni [2]? What is the advantage of CALM compared to these solutions?
2. How much data is used to train the speech generation model? Have the authors used in-house data? If not, the source of the training data should be provided.
3. As the reconstruction quality of VAE sets an upper bound for generated audio quality, it would be better if more detailed evaluation could be provided. For example, reporting additional metrics like PESQ and STOI, as well as perceptual metrics like MOS.

**References**

[1] Meng, Lingwei, et al. "Autoregressive speech synthesis without vector quantization." *arXiv preprint arXiv:2407.08551* (2024).

[2] Yuan, Ze, et al. "Continuous speech tokens makes llms robust multi-modality learners." *arXiv preprint arXiv:2412.04917* (2024).

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces Continuous Audio Language Models (CALM), a new framework for generating high-quality audio (speech and music). It addresses the primary limitation of current state-of-the-art models, which rely on converting audio into discrete tokens (using RVQ codecs). 
CALM bypasses this issue by operating directly in the continuous latent space of a Variational Autoencoder (VAE). Its architecture uses a large Transformer to model the long-term context of the audio. Instead of a slow token-by-token predictor (like an RQ-Transformer) or a multi-step diffusion process, it uses a highly efficient Consistency Model as its head. 
Experiments on both speech and music continuation show that CALM achieves higher or comparable quality to strong discrete baselines while being significantly faster and more computationally efficient at inference time.

### Strengths
1.Using a  consistency model as the head enables fast single-step generation in a continuous space. Relative to DiTAR-style “continuous latent + AR + diffusion head,” the paper’s contribution is to bring in a consistency-model-style head to reach 1–few steps and to pair this with the long/short context mechanism that specifically stabilizes music-length sequences.

2.Good performance on both speech continuation and music continuation task.

### Weaknesses
1.The paper evaluates mainly on continuation. It does not yet demonstrate text-conditioned generation, fine-grained control, or editing, where discrete-token model currently provide a convenient control surface. 

2.Asymmetric gains across domains. Music clearly benefits from “noisy long + clean short,” whereas speech shows weaker or no gains from the same trick, suggesting part of the method is tuned to music-style long dependencies rather than being universally optimal. 

3.Comparisons are mostly in-house. Many baselines are re-trained under the authors’ own setup (same data, backbone, RVQ depth). This makes the comparison clean but slightly weakens claims about superiority over the latest large-scale discrete audio LMs trained externally.

### Questions
Can you include at least one external strong discrete audio LM (not re-trained under your codebase) to demonstrate improvements are not an artifact of your reimplementation?
Since the paper currently focuses on continuation, adding a short section on non-continuation tasks would strengthen the claim that “continuous AR + consistency” is a general recipe.

### Soundness
2

### Presentation
3

### Contribution
2
