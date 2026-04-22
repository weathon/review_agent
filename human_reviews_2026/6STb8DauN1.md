# TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4, 6

## Abstract
Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human-LLM interaction. Joint text-speech modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains under-explored. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We have conducted extensive experiments to demonstrate that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. Moreover, TASTE enables straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Our experimental results show that joint modeling with TASTE outperforms other pre-trained SLMs in tasks such as speech continuation and likelihood-based next-speech selection, showcasing its effectiveness. To our best knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to learn a joint tokenization and embedding tailored for text-speech spoken language modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work proposes TASTE, a speech tokenizer that enables the alignment with semantic text tokens. It can generate highly compressed audio tokens via minimization of the speech reconstruction loss. The experiments are performed to show that TASTE successfully reconstructs the audio and helps LLMs easily finetuned to be SLMs (Spoken Language Models).

### Strengths
1. This work proposes a novel highly compressible speech tokenizer named TASTE that can reduce the modality gap between text and audio pairs. This is practically useful to reduce the length of the token sequence for a given audio with little loss of reconstruction quality. 

2. TASTE helps finetuning LLMs to SLMs via Low-Rank Adaptation (LoRA). This is practically useful to speed up the development of SLMs or omni-modal LLMs.

### Weaknesses
1. One of the most practical uses of audio tokenizers may be training of SLMs, as this work experimented. 
- Thus, it may need to be studied whether the proposed tokenizers can lead to SOTA performance of SLMs. 
- For example, TASLM’s performance can be compared with that of SOTA SLM models such as LLaMA-Omni 2. 

2. The generalization of this tokenizer should be discussed. 
- What if the domains of text and audio corpora are changed? Do we need to train again? 
- Since the proposed tokenizers are forced to learn the joint representation of both “semantics” and “phonetics”, the learned tokenizer may be much more variable according to the training sets than the one learned only focusing on “phonetics”.

### Questions
Please refer to the Weaknesses.

### Soundness
3

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
The paper introduces TASTE, a text-aligned speech tokenization and embedding method aimed at simplifying joint text–speech modeling. Given a speech–text pair, a frozen Whisper encoder provides shallow and last-layer features, an attention-based aggregator uses text tokens as queries over these features to produce a text-length speech representation, which is discretized with RVQ, and finally, a decoder reconstructs speech from the text and the aligned speech embedding for a reconstruction objective.

### Strengths
- The key contribution is to align speech tokens to the text sequence during tokenization and show that this enables simpler, stronger joint modeling at low bitrate. The idea is clean and the empirical gains on continuation with a small LM are practically meaningful for SLM research.

-  TASLM (1.3B + LoRA) outperforms prior 7B SLMs on continuation (GPT-4o, UTMOS, Human) and is competitive on SALMON/StoryCloze. The LoRA-only fine-tuning and word-level handling of ASR/LLM vocab mismatch increase practical relevance.

- ASR robustness ablations (GT vs ASR transcripts; Whisper vs Parakeet) show limited sensitivity, and the word-swap editing demo nicely evidences alignment of paralinguistics to lexical units.

### Weaknesses
- While key idea of the proposed method is clean, novelty is rather moderate. Text-speech cross-attention has been used for fine-grained TTS control, and recent joint SLMs alleviate length mismatch via interleaving (e.g., Spirit-LM) or lower-frequency tokenizers (e.g., moshi).

- Key "semantic MOS" relies on GPT-4o over ASR transcripts. Despite ablations, this double-black-box can amplify mistakes, such as prosody -> ASR error -> GPT score. Human semantic judgments or text-only LM plausibility on ASR outputs would be a stronger counterpoint, and code-switch/noisy-ASR conditions would probe robustness more realistically.

- The autogressive "weighted sum" between text and aligned speech in the unit decoder is under-specified. I would suggest the authors include equations/ablation to clarify.

### Questions
- Please detail the decoder fusion: where are the gates/weights applied between $v$ and $\hat{z}$, and how are they trained and stabilized?

- Can you report out-of-domain continuation and SALMON/StoryCloze on non-LibriVox corpora, and include dedup checks between LibriTTS and LibriSpeech?

- What are latency/memory costs for tokenization/decoding, and can TASTE stream in real time?

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
This paper introduces TASTE (Text-Aligned Speech Tokenization and Embedding), a method for learning text-aligned speech representations that directly bridge the modality gap between text and speech. The approach aggregates speech features using an attention-based mechanism guided by text transcriptions and is trained end-to-end with a speech reconstruction objective. This alignment produces compact and semantically consistent speech embeddings that preserve paralinguistic information while reducing sequence length. By adapting a pre-trained text language model with low-rank adaptation, the resulting joint model can process and generate speech coherently without additional alignment heuristics. Experiments show that TASTE improves speech continuation and next-speech prediction performance compared to existing spoken language models, demonstrating its effectiveness for unified text-speech modeling.

### Strengths
The paper tackles an important problem in spoken language modeling: how to represent speech and text within a shared embedding space for joint generation. The proposed method is conceptually simple yet effective, combining attention-based feature aggregation with an end-to-end reconstruction objective to achieve text-aligned speech embeddings. The learned representations capture both linguistic and paralinguistic information in a compact form and integrate smoothly with pre-trained text LMs through lightweight adaptation. The experimental results are consistent across multiple tasks, and the qualitative analyses illustrate interpretable alignment behavior. The paper is clearly written and provides a strong empirical demonstration of how modality alignment can improve spoken language modeling.

### Weaknesses
While the paper is technically sound, several aspects of the design and analysis remain underexplored. The attention-based aggregation is central to the method, yet its behavior and learned alignment patterns are not clearly analyzed, making it difficult to understand how alignment emerges during training. The paper lacks ablations isolating the effects of key design factors such as the chosen encoder layers, the reconstruction objective, and the LoRA adaptation on the joint model’s performance. Moreover, the evaluation focuses mainly on speech continuation and next-speech prediction, without examining general spoken understanding tasks or perceptual quality metrics that would better validate the claim of improved paralinguistic preservation. These gaps limit the interpretability and completeness of the study.

### Questions
- How exactly is the attention map used in the alignment process, and how sensitive is performance to layer selection for keys and values?
- What is the computational overhead of training the alignment module compared to standard speech tokenizers?
- Could the model be extended to use unpaired speech in a semi-supervised manner to improve robustness?
- How does TASTE handle inter-word silences or pauses during alignment and embedding generation? Are such regions ignored, or do they get represented implicitly in the embeddings?
- What is the rationale for using the last hidden layer as key and shallow hidden layers as value? Have other layer combinations been tested, and how sensitive is performance to this choice?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper presents TASTE, a text-aligned speech tokenization approach that uses transcription-guided cross-attention and RVQ to better align speech and text representations. The goal is to address the speech-text alignment mismatch common in spoken language modeling. While the motivation is clear and the design is coherent, the method largely builds upon existing codec frameworks with a text-conditioned aggregation step. Experimental results are competitive but not stronger than prior models such as SpeechTokenizer, Mimi, and EnCodec, and the evaluation omits more recent baselines. Overall, the contribution is conceptually sound but represents a modest advancement without clear empirical evidence of improved alignment and performance.

### Strengths
1. Using dynamic token frequency for better alignment with text is an interesting and well-motivated approach. It directly targets the speech-text mismatch that often limits speech tokenizer alignment.


2. Deriving speech tokens based on transcription-guided cross-attention is an interesting design choice that offers an interpretable mapping between continuous speech features and discrete text-aligned tokens.


3. The text-aligned editing demonstration is interesting. Swapping word-level TASTE tokens to transfer duration and other attributes in a controlled way is well-motivated and could inspire future work on speech style transfer or voice cloning.

### Weaknesses
1. The core idea of using text to guide lower-frequency speech tokens is sensible, but similar strategies have appeared before. Prior works have reduced speech token rates or trained tokenizers with auxiliary text objectives/distillation. The paper should better position TASTE relative to SpeechTokenizer, TWIST, DM-Codec, TadiCodec, and other text included tokenization approaches. It should clearly delineate what is new beyond the addition of a cross-attention aggregator and RVQ. The claim that TASTE is the “first end-to-end reconstruction objective to learn joint tokenization” is not fully substantiated because the distinction from prior work remains unclear.

2. The experimental evaluation is constrained. In Table 1, TASTE is compared only against SpeechTokenizer, Mimi, and EnCodec. While these are recognized baselines, they are relatively older and weaker compared to recent models such as WavTokenizer, DM-Codec, FACodec, and BigCodec, which have shown superior reconstruction performance. Similarly, Table 2 compares only to TWIST and SpiritLM, and should include newer SLMs approaches (Cosyvoice2, SparkTTS, and Llasa). Without such comparisons, the reported results lack convincing evidence of improvement. 

3. TASTE does not outperform existing baselines (e.g., SpeechTokenizer, and EnCodec) across key metrics (WER, ViSQOL, Duration Consistency, Speaker Similarity, and MUSHRA). This raises questions about the practical benefit of the more complex and resource-intensive TASTE training pipeline. The paper repeatedly attributes its lower performance to “lower bitrate,” but this argument is unconvincing without matched-bitrate experiments. The authors should directly compare TASTE with low-bitrate variants of established codecs (e.g., DAC, WavTokenizer, XCodec2, BigCodec). Without such controlled comparisons, the main claim of improved speech tokenization remains weak.

4. The design choice of setting the aggregator's keys as last-layer encoder features (K = h(L)) and values as shallow-layer features (V = h(l)) is based on questionable representational assumption and not supported with experiments. The paper assumes h(L) captures alignment cues while h(l) preserves acoustic detail, but offers no empirical analysis (e.g., attention visualizations or layer ablations) to support this. It is unclear whether the observed effects are from meaningful alignment learning or coincidental correlations. 

5. The ASR dependence of TASTE limits general applicability. The model requires accurate transcriptions to derive aligned tokens, yet there is no analysis of robustness under realistic noisy ASR or low-resource conditions. Without such tests, the practical viability of TASTE is uncertain. Moreover, The evaluation is also restricted to English corpora (LibriSpeech, Emilia). This ASR dependence also limit TASTE's applicability to spontaneous or multilingual speech. Given that TASTE explicitly depends on text-speech alignment, it is unclear whether it generalizes to languages with more complex non-phonemic orthography.

### Questions
1. Please see the Weakness section for points where additional analysis, clarification, or experimentation would strengthen the paper.
2. Can the authors provide clearer evidence that the observed improvements come from genuine text-speech alignment rather than architectural or bitrate differences? Specifically, can they (i) compare with stronger recent baselines under matched bitrates, (ii) justify the representational choice of K = h(L), V = h(l) with empirical analysis, and (iii) evaluate robustness under noisy, multilingual or imperfect ASR to confirm practical applicability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel tokenizer, called TASTE, specifically designed for text audio generative modeling. The motivation is that speech tokens generally have much longer length for the same content compared to their text counterpart. The proposed approach generates speech tokens of equal length as textual tokens. Technically, this is achieved by first passing the audio through a pre-trained speech encoder (Whisper). From this speech encoder, both last layer and shallow hidden states are extracted and fed into an attention based aggregator as key and value respectively. The queries for this aggregator are initialized by a text transcription of the speech. Since the queries are based on text, the final speech tokens match in length to the text tokens. The paper provides an intuition that these speech tokens focus on learning paralinguistic information as they are supposed to be used along with the corresponding text tokens. For decoding, a speech decoder uses both the text and speech tokens and decodes them back to speech using a vocoder. The tokenizer is trained on a speech resynthesis task wiht a cross entropy loss to predict the speech unit and commitment losses on the residual codebooks.

Apart from the tokenizer, the paper also proposes a speech language model which integrates the TASTE tokenizer, called TASLM. There are two variants - one which trains on the discrete codebook indices similar to a traditional language model and one which directly operates on the continuous latent embeddings.

Experiments are done by training on Emilia and LibriTTS datasets and evaluating on LibriTTS test split. The paper evaluates the tokenizer for speech reconstruction, speech editing and  TASLM on speech continuation, likelihood-based next-speech selection and spoken question answering.

### Strengths
- The paper proposes an interesting way to formulate tokenizers by using multimodal speech text tokens to provide complimentary information. This can help in significantly reducing the number of tokens needed to model speech.
- It is very interesting to see the behavior of this model in text-aligned speech editing. The model seems to be able to take paralingual information from one speech sequence and transfer it to another containing the same content. It would be interesting to pursue on this further and understand whether this type of editing can be done across very different voice types, like belonging to different genders etc.

### Weaknesses
- The proposed TASTE tokenizer tends to learn text aligned speech embeddings. One limitation of such an approach would be for sounds that occur commonly in speech but may not have a textual counterpart. For example, if a person laughs, it is unclear how the corresponding textual token would be retrieved. The approach may _collapse_ the laughing related information to the neighboring speech tokens if they have textual counterparts.
- The paper does not evaluate on the sWUGGGY and sBLIMP ZeroSpeech 2021 benchmark which is standard in literature [1,2]. These benchmark test the lexical and grammatical knowledge of the model to understand how well model can identify real words vs phoenetically similar non-words. I believe this might be a limitation of this tokenization approach as it needs a text transcription of the speech input.
- The paper does not explain its underperforming benchmarks. For example, In table 1, the proposed method has significantly worse WER compared to other tokenizers but the paper does not dive deeper into it. Instead, the text focuses on how WER is better than text-only baseline (343-344). Similarly, for table 2, it would be good to understand what factors contribute to the underperformance on SALMON benchmark. In table 8, this underperformance seems to happen for most of the experiments in SALMON and some discussion around it would be helpful.
- While the paper cites [2], it is unclear why it is not compared against the experiments of the paper. 
- Related work section is missing from the main paper. While it is present in the appendix, it should be a part of the main paper. 

References:
1. Spirit LM: Interleaved Spoken and Written Language Model, ACL 2025
2. Align-SLM: Textless Spoken Language Models with Reinforcement Learning from AI Feedback, ACL 2025

### Questions
- As the paper emphasizes that the goal for the speech tokens is to learn paralinguistic information, is there any sort of text encoding done to learn semantic information from the language? 
- It is unclear to me how the decoding is done from TASLM output. Since the decoder requires both text and speech, how is it achieved?

### Soundness
2

### Presentation
3

### Contribution
3
