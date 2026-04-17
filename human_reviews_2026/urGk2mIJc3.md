# SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We introduce SupertonicTTS, a novel text-to-speech (TTS) system designed for efficient and streamlined speech synthesis. SupertonicTTS comprises three components: a speech autoencoder for continuous latent representation, a text-to-latent module leveraging flow-matching for text-to-latent mapping, and an utterance-level duration predictor. To enable a lightweight architecture, we employ a low-dimensional latent space, temporal compression of latents, and ConvNeXt blocks. The TTS pipeline is further simplified by operating directly on raw character-level text and employing cross-attention for text-speech alignment, thus eliminating the need for grapheme-to-phoneme (G2P) modules and external aligners. In addition, we propose context-sharing batch expansion that accelerates loss convergence and stabilizes text-speech alignment with minimal memory and I/O overhead. Experimental results demonstrate that SupertonicTTS delivers performance comparable to contemporary zero-shot TTS models with only 44M parameters, while significantly reducing architectural complexity and computational cost.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents SupertonicTTS, a new TTS framework aimed at achieving efficient, lightweight, and streamlined speech synthesis. SupertonicTTS employs a low-dimensional latent space, temporal compression, and ConvNeXt blocks, which together reduce computational load. The model also simplifies the TTS pipeline by operating directly on raw characters and using cross-attention for text-speech alignment, thereby removing the need for G2P conversion or external aligners. Additionally, the authors propose a context-sharing batch expansion strategy that improves training stability and loss convergence speed with low resource. Experimental results indicate that SupertonicTTS achieves competitive quality compared to state-of-the-art zero-shot TTS models, despite having only 44M parameters, demonstrating its efficiency and simplicity in both design and computation.

### Strengths
The SupertonicTTS model adopts low-dimensional hidden vectors in its structural design to reduce computational complexity. It eliminates modules such as G2P and uses the cross-attention to learn the alignment between text and speech. It employs context-sharing batch expansion to achieve a larger batch size with less computational load. The article is well-structured and logically clear. It proposes effective solutions for reducing the computational cost and parameter quantity of the TTS model.

### Weaknesses
In order to enhance the streaming performance, the article removed many traditional TTS modules. However, it did not explain the impact of removing these modules on the model's effectiveness, such as issues like affecting the rhythm and pronunciations. During the process of designing the model structure for reducing the dimension of latent features, no ablation experiments were conducted to prove the benefits of such low-dimensional structural designs. During the comparison with external technical solutions, some more advanced ones were not compared, such as the FireRedTTS series which already includes FireRedTTS2.

### Questions
1. I understand that the model design of this article aims to achieve a streamlined TTS system by reducing computational load. Based on the proposed SupertonicTTS model, did the authors consider designing a pipeline that can adapt to the streaming text input and the streaming audio frame output?
2. The experimental part of the article did not include ablation experiments on the model structure design, such as abandoned G2P and reduced the latent representation. Did the authors explore the impact of such a structure design on the synthesis effect of TTS, especially rhythm? Overly simplified model design may result in a more broadcast-like effect and lack of some anthropomorphic characteristics.
3. The current advanced zero-shot technology solutions are very rich in open-source versions, such as: Cosyvoice2, Fishspeech, IndexTTS. Considering the improvement in real-time performance of the SupertonicTTS proposed in this paper, the authors can add an experimental report comparing the real-time performance.

### Soundness
2

### Presentation
4

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
The paper proposes SupertonicTTS, a compact zero‑shot TTS system with three parts: (a) a speech autoencoder that produces continuous, low‑dimensional latents; (b) a text‑to‑latent module trained with flow matching on character‑level text and a reference encoder, including two techniques, temporal compression of latents and context‑sharing batch expansion, to cut compute and stabilize alignment; and (c) a light utterance‑level duration predictor. The pipeline deliberately removes G2P and external aligners, instead relying on cross‑attention between text and reference latents. Experiments show (i) reconstruction competitive with BigVGAN and much faster decoding, (ii) batch expansion improves validation loss and WER/CER with smaller memory/time than increasing batch size, and (iii) zero‑shot pronunciation results that are competitive with (and often faster than) much larger systems while using only 44 M params and 945 h of labeled TTS data.

### Strengths
The system delivers practical efficiency at scale: a Vocos‑style autoencoder with small continuous latents and temporal compression decouples high‑rate synthesis from low‑rate modeling; the flow‑matching text‑to‑latent with cross‑attention avoids G2P/aligners and runs on raw characters; and the proposed context‑sharing batch expansion improves both loss convergence and pronunciation while being cheaper than increasing the batch size. The model is tiny (44 M total; 18.5 M in T2L) yet competitive in WER/CER versus very large zero‑shot systems, with RTF as low as 0.02/0.05 on consumer GPUs. Reconstruction is on par with BigVGAN while being >20× faster. The paper’s ablation on patching and detailed architecture docs aid reproducibility.

### Weaknesses
- Originality
    - The problem of making the TTS component lightweight and efficient has already been extensively addressed in many previous works, with various effective solutions proposed. It sounds like a problem that’s mostly been solved, which gives the impression of being somewhat outdated. Instead, it might be better to emphasize compactness, something that hasn’t been fully achieved yet, to better highlight the strength of the proposed approach.
- Quality
    - Vocos Fourier head removal: §3.1.1 replaces the Fourier head with linear layers (also in the decoder), yet no ablation quantifies quality/speed/memory vs. the standard Vocos head. Table 2 benchmarks against BigVGAN only, not Vocos itself. Please add a Fourier‑head vs. linear comparison and, ideally, a Vocos baseline under the same DSP settings.
    - Temporal compression “perfect inversion”: §3.2.1 states that compression “allows for perfect inversion.” Provide an empirical check (e.g., round‑trip latent inversion error or ABX on decompressed latents) to confirm there is no information loss at (K_c=6).
    - Reference masking vs. reference encoder: Eq. (1) masks the same segment used by the reference crop (to avoid leakage), while the reference encoder also conditions generation via cross‑attention (§3.2.3). The interaction is under‑specified; it’s unclear whether this becomes redundant or alters training/inference mismatch. Clarify when the mask applies and how inference uses the reference (and whether any reference transcript is ever required).
    - Subjective evaluation reporting: Table 6 reports mean preference scores with CIs, but the main text lacks rater selection, per‑item repeats, etc. Appx. E screenshots are helpful; please add protocol details and statistics.
- Clarity
    - The space is crowded: DiTTo‑TTS (diffusion transformer), F5‑TTS (flow matching), SimpleTTS, and VoiceBox also target simplicity or non‑AR inference. The paper compares to them but could more crisply position what is architecturally new beyond the combination of continuous latents + flow matching + ConvNeXt + no G2P/aligner. A short delta‑table would help.
    - Eq. (1) notation: briefly restate (m) (reference mask), (z_\text{ref}) construction, and whether unconditional training (CFG with (p_\text{uncond}=0.05)) omits both text and reference or either, this affects inference guidance (§3.2.4).

### Questions
- Fourier head / Vocos comparisons: Since §3.1.1 removes the Fourier head, do you have experiments showing quality/memory/speed vs. the original Vocos head?
- Temporal compression inversion: Have you empirically verified that temporal compression with (K_c=6) is invertible?
- Mask vs. reference encoder: In Eq. (1), the reference mask and reference encoder both relate to the same crop. Are they redundant? At inference, do you always provide reference audio only, or ever require a reference transcript to align text and timbre? Please clarify the exact conditioning used at inference time.

### Soundness
3

### Presentation
3

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
This paper introduces SupertonicTTS, a novel text-to-speech (TTS) system designed for
efficient and streamlined speech synthesis, which includes a speech auto-encoder, a text-to-latent module, and an utterance-level duration predictor. Experiments demonstrate a comparable performance with less parameters.

### Strengths
This paper explores several techniques to enhance architectural flexibility, improve training stability, and reduce model complexity: 
1. A remarkably low dimensionality and compress the latents along the temporal axis
2. context-sharing batch expansion to achieve the benefits of a larger batch size with minimal computational overhead.
3. employ ConvNeXt blocks

### Weaknesses
The main modules employed in this paper have been extensively explored in prior works related to speech synthesis or vocoders, which may undermine the novelty of the present work. The paper primarily focuses on optimizing training tricks.
As one of the core claimed innovations of this paper, the authors provide neither theoretical justification nor ablation studies to validate the effectiveness of using temporally compressed latent representations.

### Questions
Has the method been tested on languages other than English to verify its generalizability? If so, how well does it generalize?

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
5

### Summary
This paper presents supersonicTTS, a flow-matching-based TTS model that learns a latent representation of speech using a very lightweight 44M-parameter decoder. The model operates directly on raw text without G2P, and it folds the latent sequence along the time axis so that multiple timesteps of the latent are modeled at each decoding step, improving efficiency. The authors analyze training efficiency and pronunciation accuracy as a function of the folding factor and batch size, and they provide comparisons against zero-shot TTS models.

### Strengths
- The paper proposes a simple but cost-effective training approach: by folding along the time axis, it improves WER while increasing training efficiency.

- The decoder is extremely lightweight at only 44M parameters.

### Weaknesses
- Overall, the methodology is not particularly novel. Operating on raw text and using utterance-level duration prediction are already seen in prior work such as E2TTS [1], and the main new element seems to be the use of time-compressed latents.
- Baselines are limited and the evaluation is not well targeted. The baseline models are mostly outdated, and the objective comparison focuses almost entirely on WER, RTF, and parameter count. Widely used zero-shot TTS metrics such as speaker similarity (as proposed in VALL-E) are missing. It would be important to compare against models like ZipVoice [2], which are also lightweight and designed for fast inference with a similar purpose.
- To convincingly claim G2P-free robustness, the paper should demonstrate performance on difficult sentences and edge cases, similar to what E2TTS shows in its demos.
- At this point, it is not clear that “training quickly with a very small model” is a central research problem for TTS in the broader speech generation community, so the work may be a better fit for a speech-focused venue rather than a general ML venue.

[1] E2TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS
[2] ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching

### Questions
If the goal is lightweight TTS for practical deployment, one likely use case is streaming speech generation from an LLM. But because the architecture relies on cross-attention to the full text, doesn’t this mean the model has to wait until the entire text is available before it can synthesize, which could increase end-to-end latency instead of reducing it?

### Soundness
2

### Presentation
2

### Contribution
1
