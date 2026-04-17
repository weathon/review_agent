# PAL: Probing Audio Encoders via LLMs - Audio Information Transfer into LLMs

- Decision: Reject
- Scores: 4, 8, 4, 6, 2

## Abstract
Integration of audio perception into large language models (LLMs) is an emerging research area for enabling machine listening applications, yet efficient transfer of rich audio semantics from audio encoders to LLMs remains underexplored. The most widely used integration paradigm projects the audio encoder output tokens into the LLM input space (e.g., via an MLP or a Q-Former), then \emph{prepends or inserts} them to the text tokens. We refer to this generic scheme as \emph{Prepend to the LLM’s input token space (PLITS)} integration. We propose an efficient alternative, \underline{L}ightweight \underline{A}udio \underline{L}LM Integration \textbf{(LAL)}. LAL introduces audio representations solely via the attention mechanism within different layers of the LLM, bypassing its feedforward module. LAL encodes rich audio semantics at an appropriate level of abstraction for integration into different blocks of LLMs. Our design significantly reduces computational overhead compared to existing integration approaches. Observing that Whisper style speech encoders benefit from PLITS integration, we propose an audio encoder aware approach for efficiently \underline{P}robing \underline{A}udio encoders via \underline{L}LM (\textbf{PAL}), which in its multi encoder form employs PLITS for Whisper speech encoder and LAL for general audio encoders, and in its unified encoder form uses a single audio encoder but applies PLITS only to a compact set of speech summary tokens while integrating the full audio token sequence via LAL to preserve speech decoding capacity with low computational cost. Under an identical training curriculum, \textbf{LAL} consistently maintains performance or outperforms existing integration approaches across multiple base LLMs and tasks. For general audio tasks, LAL achieves improvements of up to 30\% over a strong PLITS baseline, while reducing memory usage by about 60\% and increasing throughput by about 190\%. Furthermore, for general audio-music-speech LLM, \textbf{PAL}, performs on par with a fully PLITS integration-based system but with substantially improved computational and memory efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper aims to improve the efficiency of information transfer from audio encoders to large language models (LLMs). It introduces LAL, which injects audio representations through attention mechanisms only, and PAL, which integrates PLITS and LAL for encoder-aware fusion.

### Strengths
1. Through architectural redesign, LAL improves both training and inference efficiency.
2. LAL demonstrates high computational efficiency compared to PLITS baselines without performance degradation, indicating potential benefits for future multimodal LLMs.

### Weaknesses
1. The advantage of LAL over existing Flamingo-style architectures is not fully convincing, as the paper lacks efficiency comparisons with Flamingo-style baselines. If my understanding is correct, the claimed improvement over Flamingo-style architectures lies in bypassing the audio FFNs to improve efficiency. However, since the main computational bottleneck in PLITS arises from audio self-attention ($O(N_a^2)$), it is unclear whether modifying the audio FFNs meaningfully contributes to reducing the overall time complexity by another $O(N_a)$. Moreover, Figure 1 only presents comparisons against PLITS baselines, without including Flamingo-style counterparts.
2. The efficiency of PAL is not reported. Figure 1 only shows comparisons between LAL and PLITS; how PAL performs relative to PLITS baselines remains unclear.
3. The criterion for selecting between LAL and PLITS in PAL is ambiguous. When the audio encoder differs from those used in the paper, it is unclear which variant (LAL or PLITS) would be more suitable or likely to perform better.
4. The writing could be improved. Section 3 (*Methodology*) mixes method and results, which should be clearly separated. In addition, the structure of PAL in Figure 2(C) is not fully explained in the text and lacks proper legends.

### Questions
Could the authors please address:

1. How efficient are LAL and PAL compared with Flamingo-style architectures?
2. Is PAL a new architecture or simply a selection mechanism that switches between LAL and PLITS depending on the encoder? In Figure 2, PAL’s structure appears different from both, yet Section 3.4 states that PAL chooses the integration per encoder between LAL and PLITS, which contradicts Figure 2. If this interpretation is correct, does it mean that when the encoder is Whisper, PAL simply reduces to PLITS?
3. In Figure 2, what do the different input square colors (purple, pink, and blue) represent?
4. In Tables 6 and 7, which models are considered fair comparisons? Some baselines outperform PAL but are not boldfaced. Does this mean they are not comparable, and if so, why? For example, Audio Flamingo-2-3B is listed in Tables 3 and 6; why is it comparable in Table 3 but not in Table 6?
5. In PAL, how should one choose between LAL and PLITS? Whisper shows better performance with PLITS, does this mean that LAL fails for speech-domain audio encoders?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the problem of efficiently integrating audio encoders into LLMs for multi-modal understanding and reasoning. the authors introduce LAL (Lightweight Audio-LLM Integration), a new mechanism that injects audio representations only through the attention mechanism of LLMs (as keys and values), bypassing the feedforward submodules entirely. This design achieves significant efficiency gains, while maintaining or improving performance compared to PLITS. Experiments across multiple LLM backbones (Llama3.2-1B/3B, Qwen2.5-1.5B) and tasks (classification, captioning, and reasoning) demonstrate that PAL achieves comparable accuracy to PLITS but with substantially lower compute and memory cost.

### Strengths
- LAL introduces a clean and elegant modification to the transformer architecture (injecting audio only via attention keys/values) offering a new efficiency–performance trade-off frontier. This idea is both conceptually simple and practically powerful.
- The authors perform extensive and controlled experiments under identical training setups, ensuring fairness. They demonstrate consistent performance improvements or parity across classification, captioning, and reasoning tasks while greatly improving efficiency.
- The introduction of PAL provides a principled encoder-aware fusion that selects integration strategies based on encoder type, offering insight into modality-specific alignment (speech vs. general audio).
- The proposed architecture achieves major computational benefits (3.5× faster training, 83% less inference memory), making it especially appealing for future resource-efficient audio LLMs.

### Weaknesses
- Since PAL’s performance depends on multiple pretrained backbones, the generalization to new or smaller encoders remains to be tested.
- The paper relies primarily on embedding-based and GPT-4–based automatic evaluation; perceptual listening or human-judged reasoning studies could strengthen claims of interpretability and semantic alignment.
- While efficiency is well-quantified, the semantic fidelity of the transferred audio information (e.g., fine-grained temporal cues) under LAL is not deeply analyzed.

### Questions
- How does the optimal layer depth for audio injection vary between encoders (SSLAM, CLAP, Whisper)? Would adaptive layer selection improve the balance between efficiency and expressivity?
- Does LAL preserve temporal ordering and sound event localization cues as effectively as PLITS, given that audio-to-audio interactions are removed?
- The results on MMAR and MMAU show parity with PLITS but not consistent improvements. Are there cases where the hybrid PAL underperforms due to conflicting encoder fusion?
- Since LAL changes the forward architecture, have you considered combining it with parameter-efficient adapters or LoRA layers for even more lightweight fine-tuning?

### Soundness
3

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
The paper proposes LAL and PAL, two efficient integration strategy to connect audio encoders with LLM. The core contribution is an architectural change called LAL, where audio tokens are only appended to keys and values in attention layers of LLM backbone. Through experiments, the authors propose to use LAL for audio encoders SSLAM and CLAP, and use traditional PLITS method for speech encoder Whisper. This hybrid strategy is called PAL. PAL significantly improves throughput and reduces GPU memory usage in both training and inference, while achieving comparable or better audio understanding performance than traditional PLITS method.

### Strengths
1. The proposed method LAL significantly boosts training throughput and reduces inference memory, while surpassing traditional PLITS method in many audio understanding and reasoning tasks.
2. The paper provides some valuable insights on how audio inputs interreacts with LLM backbone, explains how LLM digests features from different audio encoders, and proposed an encoder aware hybrid to balance efficiency and performance.

### Weaknesses
1. The idea of attending text tokens to audio tokens lacks novelty. The proposed LAL seems like a lightweight version of the "Flamingo style", where additional FFN removed and the cross and self-attention layer are combined into a single attention layer to improve efficiency.
2. The PAL seems more like a design choice based on empirical results, rather than an architectural change. The method seems more likely to be “hybrid” rather than “audio encoder aware”, since there lacks an adaptive algorithm to choose the probing method for each type of encoder. Therefore, it remains unclear if PAL can be extended to other audio encoders.
3. A typo for improved readability:
    - Line 170: the integration approach and the corresponding audio-LLM → the integration approach between audio encoders and the corresponding audio-LLM

### Questions
1. **Efficiency comparison for PAL**: Is there any comparison on throughput or GPU memory usage between PLITS and PAL rather than merely LAL?
2. **Usage of LFST**: Table 1 indicate that LFST is for achieving SOTA performance. Is LFST also applied in PAL experiments? Additionally, it would be better if results with PLITS and LFST could be provided when using Llama3.2-3B and Qwen2.5-1.5B, so that the advantage of LAL over PLITS could be evaluated under SOTA settings.
3. **Comparison with Flamingo-style**: Can LAL and PAL achieve audio understanding performance  comparable to Flamingo-style baselines? How much computational cost could be reduced? As Flamingo-style approach is closer to LAL than PLITS, it would be better if head-on comparison with Flamingo-style baselines can be included.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PAL (Probing Audio Language Models), a framework that integrates audio into Large Language Models (LLMs) efficiently. The authors aim to improve the transfer of rich audio semantics from encoders to LLMs **without incurring heavy computational costs**. PAL utilizes Lightweight Audio LLM Integration (LAL) and a hybrid integration method (PAL) that intelligently chooses the integration strategy depending on the encoder used. LAL enhances efficiency by only involving audio tokens in the attention mechanism, skipping feed-forward modules. PAL combines both LAL and PLITS (Prepend to the LLM’s input token space), depending on the audio encoder, to balance performance and computational efficiency. Experimental results show that PAL achieves better or comparable performance to state-of-the-art models, while significantly reducing **memory usage** and **computation time**.

### Strengths
1. The authors claim that PAL reduces the computational cost of integrating audio encoders into LLMs while maintaining performance compared a fully PLITS integration-based system.
- Experimental results: PAL outperforms PLITS on classification, captioning, and reasoning tasks across several LLM backbones, achieving higher throughput and lower memory usage.
- Ablation results: In Fig 1, comparing LAL with PLITS shows that LAL reduces training memory usage by up to 64.1% and increasing training speed by up to 3.5x Faster.
- **Theoretical analysis**: PAL’s design is grounded in efficient attention routing, showing how it reduces complexity and computation costs without sacrificing performance.

2. The PAL framework is evaluated on several LLMs such as Llama3.2-1B, Llama3.2-3B and Qwen2.5-1.5B. It compares LAL with PLITS in classification, captioning, and reasoning tasks on Table 1 and Table 2. It proves the generalization and scalability of PAL.

3. The paper formally proves that LAL achieves superior computational efficiency and lower memory usage compared to PLITS. To balance efficiency and performance, this paper proposes PAL that defines how audio tokens interact with text in LLM layers.

### Weaknesses
1. The authors need to further address how LAL impacts the ASR task for speech.

2. PLITS should be considered a ceiling for LAL because when all audio tokens are input together with text into the LLM, they interact through attention. However, when audio tokens are used as KV (Keys and Values) and text as Q (Query) for Cross-Attention (CA), there may be a loss of information. However, in most of the authors’ experiments, LAL outperforms PLITS. Please explain this phenomenon.

3. Comparative experiments for PLITS, LAL, and PAL in the ASR task.

### Questions
See weakness part.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PAL, a framework for efficiently integrating audio encoders with large language models (LLMs) for audio-language tasks. The authors identify two dominant integration paradigms—PLITS (prepending audio tokens to text tokens) and Flamingo-style cross-attention—and propose a novel lightweight alternative called LAL (Lightweight Audio LLM Integration). LAL injects audio representations only through the attention mechanism (as keys and values) and skips the feedforward network (FFN) for audio tokens, significantly reducing computational and memory costs. The authors further propose PAL, a hybrid system that uses LAL for general audio encoders (e.g., SSLAM, CLAP) and PLITS for speech encoders (e.g., Whisper), based on the observation that speech benefits from deeper integration. Extensive experiments across multiple LLMs and audio tasks show that LAL and PAL achieve competitive or superior performance to PLITS while offering lower memory usage and higher throughput.

### Strengths
1. LAL is a simple yet effective architectural modification that reduces both attention complexity and FFN computation for audio tokens, which is novel in the LALM training. 
2. The authors provide a principled rationale for using PLITS with Whisper (speech) and LAL with general audio encoders, supported by empirical results and a neuro-linguistic analogy. 
3. The paper conducts controlled comparisons under a standardized training curriculum, ensuring that performance differences are attributable to the integration method rather than data or model size. Evaluations span multiple LLMs (Llama 3.2 1B/3B, Qwen2.5 1.5B) and diverse audio tasks (classification, captioning, reasoning).

### Weaknesses
1. Unclear Presentation and Methodological Exposition: The paper suffers from unclear writing, particularly in Section 3.3 (Methodology), where the description of the proposed method is entangled with experimental details and results. This convoluted structure makes it difficult for the reader to cleanly grasp the core architectural innovations of LAL and PAL before being presented with empirical outcomes, hindering the understanding and reproducibility of the proposed approach.
2. Incomplete and Unconvincing Experimental Validation: The experimental validation is insufficient to robustly demonstrate the scalability and competitiveness of the proposed methods. While early results (e.g., Table 1 and Table 3) include a 3B parameter model, the final, crucial comparisons with state-of-the-art models (Tables 6, 7) are only performed with the 1B version. This raises a significant question: does the performance advantage hold when models are scaled? A fair and convincing comparison requires evaluating the 3B LAL/PAL models against other open-source audio-LLMs of similar scale (many of which have 3B variants).
3. Questionable Claims on Overall System Efficiency: The efficiency claims are narrowly focused on comparing LAL against the PLITS integration paradigm. However, the proposed PAL system employs two encoders (e.g., SSLAM/CLAP and Whisper), which inherently leads to longer input sequences and higher computational load from the encoder side. The paper does not demonstrate that PAL is more efficient than a streamlined, single-encoder model (such as Qwen-Audio or Qwen-Omni) that uses a more efficient integration method. Therefore, the claim of superior efficiency is valid only within the specific context of multi-encoder systems and may not hold against more optimized, holistic architectures.

### Questions
Refer to the weaknesses for the main concerns. The unclear presentation, incomplete and unconvincing experimental validation, and the questionable claims of efficiency lead me to reject the paper.

### Soundness
3

### Presentation
1

### Contribution
2
