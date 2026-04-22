# X-Streamer: Unified Human World Modeling with Audiovisual Interaction

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
We introduce X-Streamer, an end-to-end multimodal human world modeling framework for building digital human agents capable of infinite interactions across text, speech, and video within a single unified architecture. Starting from a single portrait, X-Streamer enables real-time, open-ended video calls driven by streaming multimodal inputs. At its core is a Thinker–Actor dual-transformer architecture that unifies multimodal understanding and generation, turning a static portrait into persistent and intelligent audiovisual interactions. The Thinker module perceives and reasons over streaming user inputs, while its hidden states are translated by the Actor into synchronized multimodal streams in real time. Concretely, the Thinker leverages a pretrained large language–speech model, while the Actor employs a chunk-wise autoregressive diffusion model that cross-attends to the Thinker’s hidden states to produce time-aligned multimodal responses with interleaved discrete text and audio tokens and continuous video latents. To ensure long-horizon stability, we design inter- and intra-chunk attentions with time-aligned multimodal positional embeddings for fine-grained cross-modality alignment and context retention, further reinforced by chunk-wise diffusion forcing and global identity referencing. X-Streamer runs in real time on two A100 GPUs, sustaining hours-long consistent video chat experiences from arbitrary portraits and paving the way toward unified world modeling of interactive digital humans.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a **dual-transformer architecture** to achieve unified multimodal understanding and generation. Specifically, the **VLM branch** handles comprehension and produces text and audio outputs, while the **diffusion branch** generates video. The two branches are temporally and semantically aligned via **cross-attention**. Additionally, the authors introduce **inter- and intra-chunk attentions** and **time-aligned multimodal positional embeddings** to further strengthen temporal and semantic coherence across modalities.

### Strengths
- The paper is **well-structured and clearly written**, with a coherent presentation of ideas.  
- It makes a **notable contribution** by introducing a unified framework for interactive human agents, which was previously handled by modular or pipeline-based systems.  
- The proposed approach shows **significant improvements** in **modality alignment**, **temporal consistency**, and **generation efficiency**, representing a meaningful step forward for multimodal generation systems.

### Weaknesses
1. According to the description, the full system seems to include **Qwen-Omni’s Talker** for audio decoding, but this component is **not reflected in Figure 2**. The authors should clarify its role and integration.  
2. As the autoregressive (AR) generation progresses, the **diffusion model’s context length** naturally grows, which could impact decoding speed. Is there **context truncation** during deployment, and if so, **how is it implemented**? Does truncation affect temporal or semantic consistency?  
3. Could a **non-autoregressive diffusion model** serve as a viable baseline for video decoding? For instance, such a model could take as input a portrait, short text, and audio segments to generate short video clips in parallel. Since the portrait remains a consistent conditioning signal, **identity preservation** might still be maintained. The paper does not seem to include a comparison to this baseline.

### Questions
See weakness above

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
3

### Summary
This paper introduces X-Streamer, an end-to-end multimodal framework for building interactive digital humans that enable real-time text-speech-video interactions from a single input portrait. It adopts a Thinker–Actor dual-transformer architecture, where the frozen Thinker interprets streaming text/audio inputs, and the Actor autoregressively generates interleaved multimodal outputs. The quantitative and qualitative results show that X-Streamer achieves strong and consistent performance in visual quality, temporal coherence, and audio–visual synchronization during real-time talking-head interactions.

### Strengths
- The paper introduces a unified architecture that seamlessly integrates text, speech, and video generation for real-time digital humans.
- X-Streamer achieves strong visual quality and maintains real-time performance.
- The paper shows stable and visually coherent long-form video generation, maintaining identity and temporal consistency over minutes of continuous interaction.

### Weaknesses
1. Limited novelty: The approach primarily constitutes an engineering integration of existing components (e.g., GLM‑4‑Voice backbone, diffusion forcing) and introduces few fundamentally new modeling ideas.
2. Insufficient experiments: The experiments on long-duration videos are limited; more cases and in-depth analyses are needed. In addition, ablation studies should examine factors such as the impact of visual context length on both performance and latency.
3. Limited scope of contribution: Although the paper frames its contribution as “human world modeling,” the demonstrated capabilities are largely restricted to audiovisual talking-head generation, which slightly overstates the scope of the claimed modeling.

### Questions
1. Please refer to points listed in weakness.
2. How does the visual context length affect performance and latency? Did the authors explore trade-offs between longer visual context for identity preservation and real-time efficiency?
3. Would jointly fine-tuning the Thinker module with the Actor improve long-term temporal consistency, multimodal alignment, or identity preservation compared to keeping it frozen?
4. Table 2 claims qualitative ablation in capiton while the content are quantitative results.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces X-Streamer, an end-to-end multimodal framework that generates synchronized text, speech, and video streams from a single portrait for digital human interactions. The framework is built upon a Thinker–Actor dual-transformer architecture. A frozen pretrained language–speech model acts as the Thinker for perception and reasoning. An Actor autoregressively generates synchronized multimodal outputs using chunk-wise diffusion forcing and global identity referencing. The framework runs in real time on two A100 GPUs, sustaining hours-long consistent video chat experiences.

### Strengths
1. Coherent framework unifying reasoning and generation across text, speech, and video for digital human interactions
2. Demonstrated cross-modal alignment and visual quality.

### Weaknesses
1. The paper's claim of real-time operation lacks substantiation, as it does not provide enough quantitative latency measurements or empirical evaluation to support this assertion.
2. Sweeping assertions such as "infinite" and "hours-long" are not backed by quantitative evidence, such as long-term consistency curves, which are necessary to validate such strong performance statements.
3. The system implementation described in the paper reads more like an engineering effort that stitches together a collection of known, established techniques and models, lacking any particularly impressive innovation.

### Questions
1. The so-called "global identity reference" is a common paradigm in LLM-based reference image/video generation, where the reference image is simply embedded as part of the multimodal input sequence. Therefore, how does your proclaimed approach substantively differ from these existing methods?
2. What is the actual end-to-end latency (ms) from user audio input to first video frame, and how are Thinker and Actor distributed or synchronized across GPUs?
3. Can X-Streamer work normally across different races, ages, genders and nonhuman or stylized avatars?
4. Can X-Streamer maintain coherence over 1-hour conversations without any degradation?
5. How to consider using a 2-second multimodal segment?

### Soundness
2

### Presentation
3

### Contribution
2
