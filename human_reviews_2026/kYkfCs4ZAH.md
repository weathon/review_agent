# FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates

- Avg Score: 5.67
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 2, 8, 6

## Abstract
Neural audio codecs are foundational to speech language models. It is expected to have a low frame rate and decoupled semantic and acoustic information. A lower frame rate codec can reduce the computational cost of speech language models by shortening the sequence length. Recent studies have developed 12.5Hz low-frame-rate audio codecs, but even lower frame rate codecs remain underexplored. We find that pushing existing audio codecs to very low frame rates loses much semantic information. We suggest that low-frame-rate codecs' limitations are in both insufficient semantic decoupling and insufficient time resolution at capturing transient phonetic details. This paper introduces **FlexiCodec** to address this limitation. FlexiCodec improves semantic preservation with a **dynamic frame rate** approach and introduces a novel architecture featuring an **ASR feature-assisted dual stream** encoding and Transformer bottlenecks.
With dynamic frame rates, it uses less frames at information-sparse regions through adaptively merging semantically similar frames. 
A dynamic frame rate also allows FlexiCodec to support inference-time **controllable frame rates** between 3Hz and 12.5Hz.
Experiments on **6.25Hz, 8.3Hz and 12.5Hz** average frame rates confirm that FlexiCodec excels over baseline systems in semantic information preservation and delivers a high audio reconstruction quality. We also validate the effectiveness of FlexiCodec in language model-based TTS. Demos are available at: https://flexicodec.github.io. Code is available at: https://github.com/amphionteam/flexicodec.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
They proposes a new codec with the following innovations:
1. frame merging based on ASR feature similarity, which leads to very low framerate codes
2. ASR feature condition, for frame merging, and also for RVQ-1

Extensive experiments show that FlexiCodec achieves good semantic and acoustic information preservation, as well as good performance in downstream NCLM-based TTS task.

### Strengths
1. Good angle: enhancing semantic info and representation, as well as low framerate compression are very important tasks
2. Good approach: leverage ASR features is probably the most straightforward approach to enhancing semantic info. Leveraging ASR features similarity for merging has a byproduct which is that we could adjust the compression rate based on the latency requirement
3. Very extensive and rigorous experiments: the authors not only did ablation studies on most design choices, standard reconstruction-based eval on WER, and acoustic info, but also shown that the alignment between dynamic merging and phonemes, as well as performance on downstream tasks TTS and audio understanding

### Weaknesses
1. XYcodec also uses ASR features, although in a slightly different way - the ASR module is finetuned during training with transcript - which will likely require more resources. Although in table 5 XYcodec is compared, I'd like to see more discussion on comparing XYcodec and FlexiCodec

2. the use of ASR features put FlexiCodec in a different category than most codec models, because it requires supervised signal. It would enhance the paper if the author can show that English-ASR trained model can also do well on unseen languages.

### Questions
Why do we use FSQ for RVQ-1, as opposed to VQ?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FlexiCodec, a novel dynamic-rate neural audio codec that successfully addresses the critical problem of semantic information loss at very low frame rates. Its core ideas are innovative and well-executed, specifically the use of a pre-trained ASR model to guide adaptive frame merging and the implementation of controllable frame rates. Supported by extensive and rigorous experiments against strong baselines, the work presents a compelling case for its effectiveness and makes a significant contribution to the field of low-bitrate speech tokenization for language models.

### Strengths
1. The core contribution of a dynamic and controllable frame rate is highly novel and effectively solves a well-motivated problem. The adaptive allocation of temporal resolution based on phonetic complexity is an elegant and powerful approach to preserving semantics.

2. The architecture is thoughtfully designed, cleverly using a pre-trained ASR model for the dual purpose of providing semantic features and guiding the frame merging process. The inclusion of Transformer modules for refinement is also a crucial detail for ensuring high quality.

3. The evaluation is exceptionally thorough and convincing. The authors compare against fairly retrained baselines, conduct extensive ablation studies that validate key design choices, and demonstrate practical utility in downstream TTS and audio understanding tasks.

### Weaknesses
1. The model's evaluation is confined to English, and its reliance on a language-specific ASR model raises concerns about its generalizability to multilingual settings without significant additional effort or resources.

2. A more detailed analysis of the codec's own computational overhead and latency during encoding/decoding would be beneficial to fully assess its efficiency.

3. The frame merging process relies on a simple, greedy left-to-right algorithm. While empirically effective, it may not be globally optimal, and a discussion of more sophisticated segmentation strategies could have strengthened the work..

### Questions
N/A

### Soundness
3

### Presentation
3

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
This paper employs a dynamic neural audio codec, which adaptively merges segments of varying lengths according to different speaking rates to achieve efficient information compression.

### Strengths
The use of ASR features as auxiliary information is well-motivated and enables more effective compression. This design allows the model to merge segments corresponding to the same phoneme, thereby shortening the overall sequence length while retaining essential information.

### Weaknesses
I am not entirely certain about my understanding of the implementation. Does each token need to maintain an additional length information field? If so, should this length information also be considered part of the compressed representation, since it seems necessary for accurate audio reconstruction? Clarifying this design choice and its impact on bitrate or compression ratio would strengthen the paper.

### Questions
How does the proposed method handle fast speech scenarios, where phonemes or syllables occur at very high rates? Can the model still achieve low compression rates while maintaining intelligibility and reconstruction quality in such cases? A discussion or experiment addressing this would be valuable.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FlexiCodec, a neural audio codec designed to operate at very low and controllable frame rates. The core contribution is a dynamic frame rate mechanism that adaptively merges semantically similar frames, guided by features from a pre-trained ASR model. This is implemented within a dual-stream architecture (semantic and acoustic) that utilizes Transformer-based modules for merging and unmerging frames. The authors demonstrate that this approach improves the preservation of semantic information. The paper also validates FlexiCodec's effectiveness in a downstream TTS application, showing competitive audio quality with substantial inference speedups in the autoregressive stage.

### Strengths
1. The idea of using pre-trained ASR features to guide the merging of semantically similar frames is intuitive and well-executed. The results in Figure 3a, showing a dramatic improvement in RVQ-1 WER at 6.25Hz, strongly validate this approach's effectiveness in preserving semantic content on in-domain data.
2. The design allows for a flexible trade-off between semantic quality and sequence length at inference time by simply adjusting the similarity threshold. This is a highly practical feature for applications with varying computational constraints.
3. The authors conduct a comprehensive set of experiments for their chosen domain, including detailed ablations of the dynamic rate mechanism (Tables 3 & 4), comparisons with numerous existing codecs, and validation on two distinct downstream tasks (TTS and audio understanding).

### Weaknesses
1. The abstract claims, "We find that a major challenge for very low frame rate tokens is missing semantic information". I do not fully agree. Recent work on syllabic / dynamic units, specifically Sylber and SyllableLM (Cho et al., 2024, Baade et al., 2024), showed that at around 6-8 Hz you can still carry the linguistic sequence reasonably well, and what starts to go missing is acoustics/prosody/fine timing, not semantics. Figure 4 of the paper actually supports a syllable-like view: FlexiCodec emits about half the phoneme rate, i.e., around syllabic granularity. The authors should tone down this claim and provide a more nuanced motivation that acknowledges that while fixed downsampling can lose transient phonetic details, the main trade-off at very low rates is often acoustic fidelity vs. semantic representation.
2. The ASR encoder (Sense Voice-Small) was trained on 300k hours of data , and FlexiCodec itself is trained on Librilight-Large (54k hours of audiobooks). The primary evaluation is on LibriSpeech-test-clean, which is also an audiobook dataset. This creates a risk that the excellent WER results are due to the ASR features being highly specialized for clean, read English speech. The claims of superior semantic preservation would be far more compelling if they were supported by an evaluation on an out-of-domain (OOD) dataset, for instance, a corpus of spontaneous or conversational speech (Emilia). This would test whether the ASR-guided merging generalizes beyond the training domain.
3. The full FlexiCodec model has 216M trainable parameters, with the Frame Unmerging Module alone containing a 100M parameter Transformer. This is a substantial model. While the paper provides Real-Time Factor (RTF) for the downstream TTS task, it omits the RTF for the codec's own encoding and decoding process. This information is critical for assessing the model's practical usability. A model that is fast for downstream tasks but slow to encode/decode may have limited applications.
4. Table 3 says: removing dynamic frame rate at 6.25 Hz increases RVQ1 WER and probing WER. That’s good, but this ablation is entangled with (a) ASR features, (b) FSQ, (c) transformer smoothing. Right now we cannot tell whether: 1) ASR features alone at 6.25 Hz already close most of the gap; 2) FSQ is the key piece at low rate; 3) or dynamic merging is the actual differentiator. I would suggest providing a factorial ablation: 1) DualCodec-style SSL features, fixed 6.25 Hz; 2) ASR features, fixed 6.25 Hz; 3)ASR features, dynamic 6.25 Hz; 4) FSQ. That way, we can see the incremental lift of each choice.
5. The paper states: “To our best knowledge, it is the one of first neural audio codecs under 10Hz… and the first work to explore dynamic frame rate on low-frame-rate neural audio codecs.” But there are concurrent <= 10 Hz speech-token systems (TaDiCodec, TASTE), the authors mention them later, and several dynamic-rate codecs, though at higher base rates. This should be toned down.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces FlexiCodec, a dynamic neural audio codec targeting particularly low frame rates (<12.5Hz) using a variable frame rate instead of a fixed frame rate, providing improvements particularly in semantic information preservation and acoustic quality. 
It uses an ASR-feature-assisted dual stream architecture to allocate more frames for complex audio segments and fewer to  information-sparse subsequences as in silences or long vowels by adaptively merging adjacent frames. 
Experiments show FlexiCodec significantly outperforms baselines in semantic preservation at 6.25Hz and 8.3Hz, while also supporting variable inference time frame rates (from 3 to 12.5Hz) to control potential trade-offs, and strong performance in downstream TTS tasks.

### Strengths
The paper presents strong empirical results and practical utility, with appropriate comparisons by e.g. retraining very recent work for lower frame rates. Previous work has primarily focused on lowering frame rates to 12.5Hz but not below. 
- A core contribution is the dynamic frame rate mechanism, which (expectedly) primarily improves semantic preservation rather than acoustic representations
  - Experiments convincingly demonstrate that this dynamic approach significantly improves semantic preservation: compare for example at 6.25Hz, a 26% relative WER reduction compared to a fixed-rate variant, and compared to recent work like DualCodec (Li et al., 2025) retrained for 6.25Hz improvements to 4.15% WER from 31.5% WER
  - This variable frame rate at inference (3-12.5Hz) is also shown to provide significant speedups for downstream TTS with reasonable tradeoffs in performance
- The novel ASR feature-assisted dual-stream architecture is also a strength - using features from a pre-trained ASR model, optimized for text prediction, for better semantic information than standard SSL features as validated by ablation studies in the appendix

### Weaknesses
- A limitation is that the decoder and downstream NAR models cannot operate directly on the variable-rate tokens: tokens must first be upsampled back to a 12.5Hz sequence via frame repetition with a (relatively) large 100M Frame Unmerging Transformer, negating some of the efficiency benefits for the synthesis stage. The Frame Unmerging Transformer, accounts for 100M of the models 216M trainable parameters
- The presented model relies on large pre-trained ASR model (a frozen 230M SenseVoice model) for the semantic features that guide the merging; other models are not compared, so it is not necessarily clear how dependent performance is on the quality and properties of this model or how generalizeable it would be to other languages or domains with weaker models

### Questions
- The Frame Unmerging Module seems relatively expensive for its task (upsampling repeated frames). Could a more efficient upsampling architecture (like a lightweight convolutional upsampler) potentially achieve similar acoustic quality? 
- Token merging is guided by the pretrained semantic features from the pretrained ASR models. The paper's notes on L430 that acoustic and semantic information density could potentially be misaligned. Could you say more about this potential misalignment and whether for example a merging criterion based on the acoustic stream could mitigate this?
- The paper notes in the limitations the model is not streaming-capable as it operates on full sequences, but that adaptations for streaming are technically. Frame merging scans from left to right to find "maximal contiguous segments", implying a need to look ahead. How would this strategy be adapted for a low-latency streaming implementation without or only a limited look-ahead?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a neural audio codec / tokenizer for English-only speech. The primary novelty is:

- the use of a specialised module to dynamically alter frame rate.
- the utilization of a pretrained and frozen ASR model as the 'coarse' token of the codec, removing the need for semantic distillation.

### Strengths
- Writing and presentation is good.
- Overall technical novelty is incremental, but the changes introduced are worthwhile and justified.
- Results seem overall fairly good

### Weaknesses
- The authors justify the focus on frame-rate by arguing that alignment with the token-rate of text is important. However, this claim of importance feels anecdotal rather than well evidenced (although it is plausible).
- (minor) The choice of 'RVQ-1' as the name for the coarse/semantic token of the stream is confusing, given that it is not produced by an RVQ (rather FSQ). I understand why the authors chose this name as there is precedent elsewhere, but a better name is needed.
- Justification of the particular pretrained ASR model used for 'RVQ-1' exists, but direct comparison with previous semantic distillation methods is absent. This makes it hard to judge the impact of this change.
- The authors state that "We have not applied FSQ for acoustic quantization because FSQ is a single-layer quantization, and we have not discovered a multi-layer FSQ practice in literature.". There is a residual FSQ formulation available in a paper you already cite - "Scaling transformers for low-bitrate high-quality speech coding" by Parker et al.
- There are many comparable baselines with public checkpoints available that are not included in the reconstruction evaluation (especially in the <0.7kbps) section.
- It's good that the authors included subjective metrics (albeit dissapointingly only for downstream TTS, not for reconstruction), but the sample size is so small that the results are very weak. The conclusions drawn from these results need to be softened greatly, given that none of the differences are statistically significant.

### Questions
- It seems 'RVQ-rest' is trained with 24 levels, and then inferenced with 8? Are all results using the truncated RVQ? Why was it trained with more levels in this case? This needs elaboration or justification.
- The design for 'RVQ-rest' utilises RVQ, but the downstream TTS models work on the continuous embeddings from this part of the bottleneck. What is the motivation for using RVQ in this section if you're not going to use the tokens?
- How is the FSQ quantizer trained? Is it using straight-through gradient estimates?

### Soundness
3

### Presentation
3

### Contribution
2
