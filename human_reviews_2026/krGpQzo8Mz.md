# Latent Speech-Text Transformer

- Avg Score: 6.00
- Decision: Accept (Oral)
- Scores: 6, 10, 6, 2

## Abstract
Auto-regressive speech–text models pre-trained on interleaved text tokens and discretized speech tokens demonstrate strong speech understanding and generation, yet remain substantially less compute-efficient than text LLMs, partly due to the much longer sequences of speech tokens relative to text. This modality imbalance disproportionately allocates pre-training and inference compute to speech, potentially hindering effective cross-modal alignment and slowing performance scaling by orders of magnitude. We introduce the Latent Speech-Text Transformer (LST), which aggregates speech tokens into latent speech patches that serve as higher-level autoregressive units. This design aligns the sequence-modeling granularity between speech and text while improving computational efficiency. The resulting patches can align with textual units to facilitate cross-modal knowledge transfer and compactly capture recurring acoustic patterns such as silence. Across story-completion benchmarks under both compute-controlled and data-controlled settings, LST consistently improves speech accuracy while also improving text performance, achieving up to +6.5% absolute gain on speech HellaSwag in compute-controlled training (+5.3% in data-controlled training). Under compute-controlled scaling from 420M to 1.8B parameters in a near compute-optimal regime, gains grow with scale, and improvements persist up to 7B parameters under fixed-token budgets. These benefits extend to downstream tasks: LST stabilizes ASR adaptation and reduces the effective autoregressive sequence length during ASR and TTS inference, lowering computational cost without degrading reconstruction quality. The Code is available at https://github.com/facebookresearch/lst.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes the Latent Speech‑Text Transformer (LST), which models patches of speech tokens instead of individual tokens to reduce the speech‑vs‑text compute/sequence‑length imbalance in interleaved speech‑text LMs. A lightweight local encoder/decoder forms and reconstructs speech patches, while a global transformer models interleaved text tokens and speech patches; alignment‑based patching (Wav2Vec2+CTC word boundaries) and curriculum patching (aligned→static) are introduced to better synchronize content while enabling simple, static‑only inference. Across compute‑controlled and data‑controlled protocols, LST improves both S→S and T→T on HellaSwag, StoryCloze, TopicStoryCloze, with clearer gains from curriculum patching and competitive compute savings; the method also scales from 1B→7B parameters with consistent improvements.

### Strengths
LST is a practical, well‑motivated way to mitigate speech‑text length and compute imbalance: it pairs a local encoder/decoder (restricted windows) with a global transformer to operate on information‑dense speech patches; alignment‑based patching improves semantic synchrony while curriculum avoids reliance on aligners at test time. The paper executes careful compute‑controlled and data‑controlled comparisons showing consistent S→S and T→T gains, compute savings under a fixed speech/text token budget, and robust scaling from 1B→7B. The patching‑strategy ablation clarifies why curriculum starting from Align (sil sep.) works well.

### Weaknesses
- Originality
    - The novelty is largely an application of BLT‑style patching to the speech‑text setting; stronger baselines (e.g., late cross‑attn fusion, gating/MoE variants) are not contrasted head‑to‑head under the same compute.
    - Alignment‑aware and curriculum schedules are natural extensions; their conceptual leap is moderate relative to prior patching ideas in other modalities.
- Quality
    - Aligner choice is fixed to Wav2Vec2+CTC; there is no comparison to Whisper forced alignment or MFA‑style alternatives, so the robustness of the alignment dependency is unclear.
    - The BPE SpeechLLM baseline uses 1k SentencePiece trained on 100k speech sequences, with a note that 5k/10k didn’t help; however, the paper doesn’t show tokenizer sufficiency checks (e.g., stability/perplexity/segmentation quality) that would rule out an under‑trained BPE baseline.
    - Curriculum vs. Align (sil merged): Table 6 shows Align (sil sep.) generally stronger than sil merged on S→S, but it’s not shown whether a curriculum initialized from “sil merged” could match or exceed “sil sep.” for specific patch sizes.
    - Compute savings are reported, but a concise methodological paragraph clarifying how savings are measured versus baseline token→patch conversion would aid interpretation.
- Clarity
    - Figure 2 terminology: “Patch Encoder/Decoder” appear to correspond to §3’s Local Encoder/Decoder; making this mapping explicit in the caption/text would reduce confusion (Fig. 2; §3).
    - Local Encoder cross‑attention: clarify what queries what. Do latent patch embeddings query a local window of speech token embeddings (keys/values)?
    - Dataset licensing/availability: Spotify Podcast (55k hrs) is listed (Table 1); please state license and present‑day access status for reproducibility/ethics.
- Significance
    - The gains are solid on story completion tasks synthesized with Kokoro TTS, but the generalization to other speech understanding tasks is not explored; a short discussion of when patching helps most would strengthen the takeaways.

### Questions
- Can you confirm that Patch Encoder/Decoder in Fig. 2 are exactly the Local Encoder/Decoder of §3 and make that explicit in the text/caption? Also, in the local cross‑attention, what is the query and what are the keys/values?
- Did you compare Wav2Vec2+CTC to Whisper alignment or MFA for word/BPE boundaries in terms of accuracy, speed, and downstream impact on LST vs. baseline? If not, any evidence that W2V2+CTC is near‑optimal for your training/inference regimen?
- For BPE SpeechLLM, how did you determine the SentencePiece tokenizer is sufficiently trained (beyond vocab size sweeps)? Could a better BPE (more data, longer training, different seed, or unigram LM) close the gap to LST? Please provide tokenizer diagnostics or an expanded ablation.
- You start curriculum from Align (sil sep.). Did you try starting from Align (sil merged) (arguably closer to static evaluation) and, if so, how do results compare across patch sizes?.
- Given Spotify (55k hrs) is a substantial portion of training, can you clarify license, current accessibility, and whether research‑only use is still permissible? A short line in §4.1/Ethics would help replicability.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper addresses a significant and well-known challenge in auto-regressive speech-text models: the information density mismatch between modalities. Speech, when tokenized (e.g., using HuBERT), results in disproportionately long sequences compared to the equivalent text tokens. The authors hypothesize this mismatch hinders speech-text alignment and leads to poor computational efficiency and scaling laws. To solve this, the paper introduces the Latent Speech-Text Transformer (LST), an architecture inspired by the Byte Latent Transformer (BLT). The core idea is to aggregate sequences of speech tokens into "latent speech patches" using a patch encoder. A global transformer then processes a shorter, more balanced sequence of interleaved text tokens and these latent speech patches. A patch decoder maps the latent representations back to speech tokens for generation. Moreover, the proposed LST architecture, and especially the curriculum patching method, is an innovative, practical, and effective solution. It demonstrates state-of-the-art performance gains over strong baselines in a rigorously controlled and very convincing experimental setup. Moreover, the work provides a practical and scalable architectural solution that improves the efficiency and alignment of multimodal speech models, which is of broad interest to the ICLR community.

### Strengths
- The paper's primary strength is its direct and effective approach to a clear and significant problem in speech-text modeling. 
- The motivation is well-articulated, and the proposed LST architecture is a logical adaptation of patching techniques from other domains. The experimental design is rigorous, particularly the use of both compute-controlled and data-controlled settings, which provides a robust validation of the method's efficiency gains. 
- The most substantial contribution is the curriculum patching strategy. This is an innovative and highly pragmatic solution that achieves the "best of both worlds": it leverages the rich semantic guidance of an external aligner during training to produce robust representations, but transitions to a simple, dependency-free static patching method for inference. 
- The strong and consistent performance gains of this curriculum-based model over all baselines, combined with promising scaling results up to 7B parameters, makes it very effective.

### Weaknesses
- The evaluation is focused on high-level narrative and commonsense reasoning tasks, with fine-grained lexical and syntactic benchmarks like sWUGGY and sBLIMP explicitly omitted. 
- This leaves open the question of how LST's token aggregation might affect performance on tasks requiring very fine-grained acoustic-phonetic or syntactic judgments. 
- Furthermore, the provided ablation on LST (Static), which is the "pure" architecture without aligner supervision, shows mixed results: it demonstrates a clear and strong advantage over both baselines on HellaSwag but underperforms the base model on TopicStoryCloze. This inconsistency, however, serves to strengthen the paper's main claim by justifying why the novel curriculum patching approach is necessary and superior, as it successfully resolves this instability and delivers robust performance across all tasks.

### Questions
I would appreciate clarification on the following points:
- The authors mention omitting sWUGGY and sBLIMP. While I understand the focus on narrative reasoning, could you speculate on how LST's aggregation mechanism might perform on these tasks? Would the patch decoder be sufficient to reconstruct the necessary fine-grained information, or do you expect a trade-off?
- How sensitive is the curriculum patching method to the quality of the Wav2Vec2+CTC aligner? Would the benefits be held if a simpler, less accurate, or different-style aligner were used during training?
- The t-SNE plots in Figure 4 are a nice qualitative illustration of LST's alignment. This claim would be significantly strengthened by including a parallel t-SNE plot showing the speech token embeddings from the baseline model for the same words. A direct visual comparison of cluster separation and tightness would make the "improved alignment" point much more immediate and convincing.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the latent speech-text transformer, a pre-training method for speech-text language models, with the goal of reducing the computation required for long speech sequences and the sequence length mismatch between text and speech. Specifically, the model works by aggregating discrete speech tokens into latent patches.The authors train LST models on interleaved speech-text data and ablate the different patching strategies used. Overall, LST models perform better than simple speech-text LMs on textual and spoken versions of commonsense and narrative coherence benchmarks.

### Strengths
- The paper is well-written and easy to understand.
- The latent speech-text transformer (LST) is a novel architecture that attempts to address the long sequence length of speech tokens and the length mismatch with text. This is a step towards solving a significant problem that makes scaling speech models difficult
- The authors propose and benchmark a variety of patching techniques that aggregate speech tokens together, and show how they can be combined into a stronger and faster model through curriculum learning
- They analyze several factors related to the proposed method, such as scalability and compute equivalence.

### Weaknesses
- Unclear evaluation procedure: the chosen evaluation sets rely on multiple choice QA. This means that in the speech setting, the model outputs HuBERT tokens. How are the output tokens converted to the actual multiple choice answer? Its unclear if ASR or some other method is used to map the model output to the actual answer choice. Without such information, reproduction and fair comparisons against this work are challenging.
- Small test coverage: the evaluation only covers the aforementioned mQA tests. While this is a good evaluation of the model's "intelligence"-related capabilities, I am surprised there was no evaluation on more traditional speech tasks like ASR or TTS that measure the model's phonetic and cross-modal capabilities. I believe that it would be vital to test such abilities, since they may be affected by the compressed representations, which would lower the impact of the proposed method. Such experiments are done in the SpiritLM paper, which can be compared to.
- While fine for understanding tasks, the proposed technique would be non-trivial to extend the multi-stream speech LMs that use neural codecs, which are the SOTA for generation tasks, limiting its impact.
- While the scaling figure is nice, I think the data points appear too close together to be very meaningful (only from 10K to 25K iterations) for models that are trained for 200K iterations.

### Questions
- How are the output hubert tokens converted to the actual multiple choice answer? Its unclear if ASR or some other method is used to map the model output to the actual answer choice.
- Do they spoken Hellaswag / SC / TC datasets have multiple speakers or only use a single speaker?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes _Latent Speech-Text Transformer_ (LST) for improving speech LLMs by using a pair of local encoder/decoder similar to the [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453.pdf):
1.  The local encoder reduces the speech token rate via a cross attention layer turning speech tokens into _patches_.
2.  The speech patches along with text tokens are processed by the global transformer. 
3.  The output from the global transformer can be used to directly predict the next text tokens. In order to predict the speech tokens, the corresponding global transformer output tokens are fed to the local decoder, which restores the tokens into the original speech token rate.

Three patching schemes are explored:
-   Static patching: Each patch consists of a fixed number of speech tokens without any overlap.
-   Alignment patching: Each patch consists of a word / silence obtained from forced alignment using Wav2Vec2-CTC.
-   Curriculumn patching: Training starts with alignment patching, then gradually transitions to static patching.

Models trained from scratch using text and speech datasets are evaluated against a baseline transformer model that directly processes speech tokens in the same manner as text tokens. Evaluations are conducted on HellaSwag, StoryCloze, and Topic StoryCloze. For evaluating the speech processing capability of the proposed model, these test sets are also TTS'd by the authors using Koroko TTS. Evaluation results show LST leads to a noticeable improvement in both text and speech versions of these test sets.

### Strengths
- Originality: This paper applies Byte Latent Transformer, originally designed for better byte based language modelling, to a new problem of speech language modelling. This is a reasonably novel approach compared to other approaches such as low bitrate speech codecs (e.g. SpeechTokenizer or encodec). The curriculum learning scheme that transitions from alignment patching to static patching is also novel.
- Quality: The use of latent transformer is a simple and well motivated method for reducing the speech token rate, eventually resulting in a better language model for both speech and text.
- Clarity: Most part of the paper is well organized and clearly written.
- Significance: The proposed method compares favorably against the baseline model in both the text and TTS versions of HellaSwag, StoryCloze, and TopicStoryCloze evaluations.

### Weaknesses
-   Clarity: To help readers not already familiar with Byte Latent Transformer, the description of the local encoder/decoder (161-185) could use some expansion. This is central to the main idea of this paper, and thus it would be great if the readers do not have to refer to another paper to understand the core ideas of this paper.
-   Quality: Several changes in evaluation & experiments would be necessary to better support the claim of this paper.
    -   In terms of speech modelling, all the evaluation test sets in this paper are produced by TTS. There is no evaluation on actual human speech, which often differ greatly from clean synthesized speech. For comparison, [the Spirit LM paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00728/127457), which is this paper's baseline model, included results on ASR (Table 5) and a comparison against a cascade system.
    -   It is not clear why the authors chose to evaluate on their version of synthesized StoryCloze/TopicStoryCloze while there exists [a widely used version from the original authors](https://github.com/slp-rl/SpokenStoryCloze). This makes comparison against results from other papers very difficult.
    -   The model is supposedly capable of both understanding and generating speech (line 269), however this is only evaluation of speech understanding in the current draft. In constrast, there is TTS evaluation in the Spirit LM paper.

### Questions
Could you clarify what "compute-controlled" means in line 321? I'd expect the baseline model to need much more compute to process the same speech input due to the higher number of HuBERT tokens and the quadratic time complexity of transformer. But Table 3 seems to suggest that the number of interleaved tokens is only slightly lower different in the baseline compared against LST. In a setting where the baseline uses roughly the same flops, shouldn't it see far fewer speech tokens?

### Soundness
2

### Presentation
2

### Contribution
2
