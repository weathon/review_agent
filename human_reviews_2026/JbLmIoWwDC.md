# Gogo: Group-wise granularity-ordered codec for stable and efficient speech generation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Current speech language models require their core component, the speech codec, to discretize continuous speech signals into tokens that not only capture high-level cues for autoregressive modeling but also preserve sufficient acoustic details for perceptual quality. To address this need, we propose Gogo, a group-wise granularity-ordered codec that quantizes each group of frames into tokens arranged from coarse to fine, where coarse tokens encode high-level abstractions and fine tokens progressively recover low-level details. Building on the granularity-ordering property of Gogo, we introduce GogoSpeech, a two-stage speech language model that performs speech generation by first constructing a coarse speech backbone at an extremely low token rate and then enriching the backbone with fine-grained acoustic details. Considering the inherently non-uniform information distribution in speech signals, we further design a Group Relative Policy Optimization (GRPO)-trained token allocator that adaptively allocates token budgets to groups based on group-wise complexity. Experimental results demonstrate that Gogo delivers state-of-the-art reconstruction performance across most metrics at a token rate of 47. Moreover, evaluations on zero-shot text-to-speech tasks show that GogoSpeech enables efficient generation by adaptively reducing the average token rate, and attains state-of-the-art results in long-form speech generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors present Gogo, a group-wise granularity-ordered speech codec. The proposed method tokenizes contiguous frames into groups and sequentially orders tokens from coarse to fine granularity. Utilizing Gogo codec, the authors construct GogoSpeech, an LLM-based text-to-speech (TTS) model. GogoSpeech synthesizes speech in two stages: the first stage generates high-level acoustic cues ("speech backbone") at a low token rate, and the second stage refines these cues by adding detailed acoustic information at a standard token rate.

### Strengths
Overall, the proposed approach is interesting. GogoSpeech achieves performance comparable to SOTA TTS models.

### Weaknesses
(1) Paper Organization\
The main text lacks essential ablation studies and related work sections. The current paper structure does not sufficiently support the primary claims regarding the effectiveness of the proposed approach. I this this paper requires major revision to clearly emphasize core contributions and key experiments in the main body, while relegating supplementary details to the appendix.

(2) Questionable Impact of Asymmetric Masking\
Appendix Table 6 indicates that removing asymmetric masking leads to negligible performance differences (e.g., WER increases from 2.394 to 2.406, indicating no significant difference, described by the authors themselves as "slight"). Given that asymmetric masking is presented as a major component in Section 2.1.3, this result unfortunately undermines its claimed effectiveness. Among the three key techniques introduced in Section 2.1.3, the ineffectiveness of asymmetric masking diminishes the overall contribution.

(3) Insufficient GRPO Details and Experiments\
The authors mention using a "slightly modified GRPO," specifically the removal of the KL penalty and exhaustive enumeration of token budgets. However, critical ablation studies investigating these specific modifications and sensitivity analyses concerning the reward weights are missing. Furthermore, comparisons to alternative reinforcement learning-based objectives (e.g., DOP) are also absent. Such analyses are crucial to substantiate the efficacy and necessity of the proposed GRPO-based token allocator.

(4) Fairness and Depth of Zero-shot TTS Comparison\
In Table 3, GogoSpeech (WER: 2.394) achieves competitive but not consistently superior performance compared to state-of-the-art methods such as F5-TTS (WER: 1.830). Although GogoSpeech slightly outperforms F5-TTS in long-form generation, the overall comparison remains mixed and inconclusive. Due to substantial differences in architecture and training conditions, this heterogeneous comparison does not clearly demonstrate the efficacy of the proposed granularity-ordered tokens.

To rigorously isolate the impact of the proposed approach, systematic experiments varying frame rates, token rates, and codebook sizes under consistent training conditions are required. While claiming state-of-the-art performance would be important, systematic evaluations provide stronger evidence than heterogeneous comparisons. Additionally, if the authors wish to claim only state-of-the-art results, comparisons with API-based TTS systems (e.g., ElevenLabs TTS, OpenAI TTS, Gemini TTS) would significantly enhance the evaluation's rigor and relevance.

### Questions
I have no questions but encourage the authors to improve their presentations and experiments.

### Soundness
3

### Presentation
1

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
This paper proposes a new speech codec that groups the mel feature frames into groups and maps them to coarse-to-fine scale tokens. The token sequence then go through the two-stage GogoSpeech model and the outputs are decoded by a flow matching model into mel features after which a Vocos vocoder is used to get the wave file. The codec tokens are learned by some query vectors and the model is trained so that the more frequent and more salient features are expressed in the first few tokens and the granularity increases as the number of tokens representing a speech signal increases. The novelty lies in the design of the codec, and the use of some experimental ideas such as asymmetric masking, nested dropout, a token length based loss balancing terms. The model is trained on the Emilia dataset. Llama is used as the LLM backbone for the autoregressive modeling of the tokens. Experiments compare different codecs with the Gogo codec, where the proposed system achieves competitive performance to other codecs. For codec operating at a similar rate, Gogo slightly outperforms the other ones in terms of PESQ narrowband, speaker similarity, and WER. GogoSpeech TTS model also performs better than the baseline TTS models in terms of subjective MOS scores. On objective metrics (WER and SIM), GogoSpeech either slightly outperforms or is on par with the other TTS models mentioned in the paper.

### Strengths
originality, 
+ The paper proposes a new speech codec that can represent speech at various granularity. 

quality, 
+ Experiments demonstrate positive results at a relatively small token rate. 

clarity, 
+ Mostly clearly written, there are extensive appendices that provide further details. 

significance
+ Because of the introduction of a new speech codec, the paper is relevant to the speech community as well as the speech + text joint LLM modeling community as most joint models utilize a form of speech codec.

### Weaknesses
- The paper introduces a new codec and experiments positively demonstrate the effectiveness of the codec in audio reconstruction and TTS. As mentioned in the introduction, LLM that accept speech inputs usually make use of a speech codec to tokenize the speech input and hence the paper is relevant to speech LLMs. However, to get a full picture of how this codec would work in a speech + text joint LLM setting (e.g. on a spoken QA task), it could have been better if some further experiments had been included. 

- The paper also analyzes what the tokens at different levels learn, which is good. However, it might have been more interesting to see the codec perform in a task where the factors listed in Figure 5 (such as jitter, shimmer, etc.) would more explicitly show its use case. These tasks could have been speech emotion recognition, identification of speech attributes, etc. instead of formulating the probing task as a regression task.

### Questions
1. Do the authors think that speech reconstruction and TTS experiments are sufficient to prove that the proposed codec can be successfully utilized in most multimodal LLMs (particularly speech and text LLMs)?

2. Could you please clarify how the regression probing task is set up? Is it based on masking out certain tokens and reconstructing the same audio? 

3. In Figure 1, there is a mention of a token budget. Could you please clarify whether once this token allocator is trained, we get the same number of tokens per group? Or is it dynamic within an utterance, i.e., each group getting different number of tokens?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces Gogo, a new speech tokenizer, and GogoSpeech, a two-stage autoregressive text-to-speech (TTS) framework built on top of it.

The Gogo tokenizer consists of a Transformer encoder that processes grouped mel-spectrogram frames augmented with learnable query tokens, and a flow-based generative model for mel-spectrogram reconstruction. The encoder outputs corresponding to the queries are quantized using Finite Scalar Quantization (FSQ), and combined with filler tokens as conditioning input for the flow model, followed by a pretrained vocoder to synthesize the waveform. To structure token learning, the model enforces a coarse-to-fine hierarchy across the query tokens through asymmetric masking, nested dropout, and adaptive loss balancing, encouraging early tokens to capture global content while later tokens refine acoustic details.

Building on this tokenizer, GogoSpeech employs a two-stage generation process: the first stage predicts a coarse speech backbone from text, while the second stage adds fine-grained acoustic details. Additionally, a reinforcement-learned token allocator dynamically determines the number of fine tokens to generate per segment, improving efficiency by reducing the average token rate from 47 Hz to around 35–36 Hz. Experimental results demonstrate the effectiveness of the proposed framework.

### Strengths
I like the idea of using learnable queries to capture the information from grouped mel-spectrogram frames — it’s a neat and flexible way to summarize local acoustic context. Grouping the frames itself is also smart, as it naturally reduces the token rate without much loss of information.

The hierarchical design in the query tokens is particularly interesting. Through asymmetric masking, nested dropout, and adaptive loss weighting, the model explicitly encourages a coarse-to-fine structure across tokens. The probing results in Figure 5 clearly reflect this — earlier tokens capture higher-level linguistic features, while later ones focus more on detailed acoustic attributes like pitch and spectral characteristics.

The addition of the token allocator is another strong point. It’s an elegant way to make the system adaptive by deciding how many fine tokens to generate per segment, which not only saves computation but also reduces the overall token rate without hurting quality.

### Weaknesses
1. The paper does not discuss the limitations of the proposed methods, such as potential trade-offs in efficiency, reconstruction artifacts, or scalability.  

2. The use of filler or placeholder tokens for the flow-matching decoder may lead to hallucinations or slight pronunciation errors during reconstruction. For example, in the Codec Comparison samples, the first utterance (“he”) sounds closer to “the,” suggesting occasional contextual confusion.  

3. The codec architecture involves three major components — a Transformer encoder, FSQ quantizer, and a flow-matching decoder — where the flow-based component requires iterative inference. This could make the overall reconstruction process slower than other non-iterative codecs.  

4. GogoSpeech employs two fully autoregressive stages (for backbone and fine-detail prediction), both predicting one token at a time. Since these are cascaded, inference time can be considerably higher than single-stage or non-autoregressive systems.  

5. There is no evaluation or comparison of runtime efficiency, inference speed, or computational cost against existing baselines, which makes it difficult to assess the practical performance of the system.  

6. Details of the subjective evaluation are missing. The paper does not specify the number of evaluators, number of test samples, listening protocol, or any statistical testing, which limits the reliability of the perceptual results.

7. Statistical significance analysis is not provided for any of the reported metrics. Including confidence intervals or significance tests would help clarify whether the observed improvements are meaningful.

### Questions
1. Could the authors clarify how the 4-dimensional grouped input to the Transformer encoder is handled in practice? Specifically, is the `n_g` (group index) dimension flattened into the batch dimension before encoding?  

2. Regarding the autoregressive nature of GogoSpeech, the math in Section 2.2 suggests token-level generation within each group. Could the authors confirm whether the model predicts one token at a time across all groups, or all tokens of a single group before moving to the next? It is also unclear how the “flatten” operation affects the sequence length (e.g., whether it becomes *T × b*).  

3. The demo or samples page does not load properly — I was only able to access the codec comparison samples, not the others.

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
4

### Summary
This paper proposes Gogo, a novel group-wise granularity-ordered speech codec, and GogoSpeech, a two-stage speech language model for generation. The core idea is to quantize groups of frames into tokens ordered from coarse (high-level information) to fine (acoustic details). GogoSpeech leverages this by first generating a low-rate "speech backbone" (coarse tokens) and then enriching it with fine details in a second stage, aiming to improve stability and efficiency. Additionally, a GRPO-trained token allocator is introduced to adaptively assign token budgets to different speech segments based on their complexity, further enhancing efficiency. Experiments show that Gogo achieves strong reconstruction performance, and GogoSpeech obtains good results in zero-shot TTS.

### Strengths
Novelty: The core idea of group-wise, granularity-ordered tokenization is novel and well-motivated. It presents a promising alternative to the dominant frame-wise paradigm.
Systematic Design: The system is thoughtfully designed. The GogoSpeech model is built logically upon the properties of the Gogo codec, and the token allocator further enhances the system's efficiency.

### Weaknesses
Modeling Frame Rate: The final token rate of 47 Hz (or 36 Hz with the allocator) is presented as efficient. However, this claim is weakened when compared to recent models that operate at much lower rates, such as 12.5 Hz or even 6.25 Hz. The paper does not adequately position its efficiency gains against these highly competitive low-bitrate models.
Lack of Direct Computational Metrics: The paper relies on token rate as a proxy for efficiency. This is insufficient for a fair comparison. Crucial metrics like Real-Time Factor (RTF) on a standardized hardware setup or computational complexity (e.g., GFLOPS) are missing. Without these, the claim of "efficient generation" remains largely unverified.
Unfair Baseline Comparison in TTS Evaluation: In Table 3, the proposed English-only model is compared against several multilingual TTS systems (e.g., CosyVoice 2). This is not a fair comparison, as multilingual models need to handle the complexities of multiple languages, which can compromise performance on any single language compared to a specialist model. This limitation must be explicitly acknowledged and discussed.

### Questions
Could you please provide direct inference efficiency metrics such as RTF (with hardware specifications) or GFLOPS to allow for a more direct and fair comparison of GogoSpeech's computational cost against other models? How does the overall efficiency compare to models with much lower token rates (e.g., 12.5 Hz)?
Regarding the TTS comparison in Table 3, can you comment on the fairness of comparing your English-only system to strong multilingual baselines? Could the comparison be strengthened by evaluating against English-only configurations of these models, if available?
The token allocator is an interesting component. How do you expect it to perform in a multilingual context? Would different speaking habits, prosody, or phonetic structures across languages (e.g., tonal vs. non-tonal) affect its ability to accurately assess group complexity, or would it generalize well without language-specific fine-tuning?

### Soundness
2

### Presentation
3

### Contribution
2
