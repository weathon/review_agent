# AlignChat: Endowing LLMs with End-to-End Speech-to-Text Chat Capability through Token-Level Representation Alignment

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 2, 2, 6

## Abstract
The advent of large multimodal models (LMMs) such as GPT-4o has intensified interest in equipping large language models (LLMs) with end-to-end speech understanding capabilities. Existing methods typically employ encoder-based audio tokenizers to map speech into audio tokens served as LLM inputs; while effective, the frequency discrepancy between audio and text tokens demands large quantities of speech data and costly LLM finetuning to achieve cross-modality alignment, while potentially harming the original capability of the LLM backbone. In this work, we introduce *AlignChat*, a simple yet effective framework that bridges speech and text modality via a speech tokenizer with encoder–decoder-based Transformer architecture, ensuring precise one-to-one token-level alignment and efficient cross-modality knowledge transfer without the need for finetuning the LLM backbone. AlignChat adopts a two-stage training scheme. The computation-efficient pretraining stage only requires the speech tokenizer and embeddings of the LLM for preliminary cross-modality alignment, while the instruction-tuning stage proposes to use self-generated speech-instruction-response pairs to ensure consistency between speech- and text-conditioned behavior of AlignChat. Experiments demonstrate that AlignChat achieves strong performance on speech-to–text chat benchmarks with only ~1/20 of the speech data used by previous methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors try to address the resolution mismatch issue among speech and text token representation in speech LLM. They propose to use AED based framework to synchronize the speech token with text token. Two stages training is employed with ASR pre-training and instruct fine-tuning. The results show good performance on 3 benchmarks. 
My main concern is novelty. The proposed method is quite similar to BESTOW [1] but with a more powerful LLM and better results on three speech benchmarks. Also, the comparison in this paper is unfair. The proposed system can only generate text modality but most systems compared in this work could generate both text and speech outputs.

[1] Chen, Zhehuai et al. “Bestow: Efficient and Streamable Speech Language Model with The Best of Two Worlds in GPT and T5.” SLT (2024)

### Strengths
The authors build a good system with the proposed method, which shows good performance on 3 open-source benchmarks.

### Weaknesses
- The authors consider the resolution mismatch of speech token and text token could hurt the performance and propose to use AED based framework to integrate speech token into the LLM. However, the work doesn't give direct evidence to support this claim. It would be great if authors could conduct similar comparison similar as [1] to give more evidence. Also, the ablation study in table 3 also shows that extra decoder only provides small gain compared with the AED framework.
- The proposed method is quite similar to BESTOW and no discussion is provided in the related work section.
- The proposed system is not able to generate audio compared with other baseline systems in Table 2. 


[1] Lam, Tsz Kin et al. “Prepending or Cross-Attention for Speech-to-Text? An Empirical Comparison.” NAACL (2025)

### Questions
- How does this method compared with a fully cascaded approach, i.e., ASR (Whisper)  and LLM (Qwen2.5-7B-Instruct)? 
- What's the impact of modality projector given the alignment and ASR training on the decoder?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an alignment framework between acoustic embeddings and corresponding LLM-based token embedding. The authors utilize an attention-based encoder-decoder (AED) ASR model (Whisper) to extract acoustic embeddings from the decoder which is aligned against the LLM embeddings. AlignChat achieves this through a two-stage process: first mapping speech representations into the LLM’s embedding space, then instruction-tuning the model to follow prompts directly from audio. By freezing the LLM and introducing only a lightweight speech adapter, the method preserves the model’s reasoning ability while enabling end-to-end speech interaction. Experiments show that AlignChat performs competitively on speech dialogue benchmarks with a fraction of the training data used by prior systems.

### Strengths
Strengths:

1. The paper is well written and easy to follow.
2. The paper tackles an important problem.
3. The problem formulation is expressed clearly and is mathematically sound.
4. Overall impressive results for speech-to-text chat tasks.
5. Comprehensive analysis in the form of ablation studies and visualization of learnt representations.

### Weaknesses
Weaknesses:

1. This paper misses out on citing some key initial papers on token-level cross-modal embedding alignment. For example,

[1] Kubo et al., "Knowledge transfer from large-scale pretrained language models to end-to-end speech recognizers"

[2] Choi et al., "Distilling a pretrained language model to a multilingual ASR model"

[3] Sunder et al., "Tokenwise contrastive pretraining for finer speech-to-bert alignment in end-to-end speech-to-intent systems"

[4] Sunder et al., "Fine-Grained Textual Knowledge Transfer to Improve RNN Transducers for Speech Recognition and Understanding"

While these paper were before the LLM era, the underlying philosophy of embedding alignment is exactly the same and hence they merit to be acknowledged.

2. Some missing experiments/discussion about cascaded v/s end-to-end approaches (see next section).
3. Lack of clarity in the design of $L_{align}$ as a combination of $L1$ loss and cosine similarity loss (see next section).
4. Please see next section for other questions and comments.

### Questions
1. The fact that the authors used the decoder output of an AED ASR model as the acoustic embedding for alignment and subsequent input to the LLM raises some questions:

a. How do the authors deal with the mismatch during training and inference? During training, ground-truth transcripts are available, therefore, there is indeed a one-to-one correspondence between the acoustic and text embeddings. However, there will always be errors during inference, in particular, insertion and deletion errors which prohibit this alignment. How is this taken into account.

b. The authors suggest that they use greedy decoding during inference from the AED model. How does this compare with beam search decoding apart from the obvious latency differences? 

c. How is $L_{speech} = L_{text}$ (line 139) ensured during test time?

d. The authors mention in line 264, "During inference, speech inputs are first encoded into audio features and then autoregressively converted into speech representations that maintain a strict correspondence with text tokens". That cannot be true in the presence of ASR errors. Please re-phrase.

e. Finally, I don't see baseline numbers where Whisper was used as a first-pass ASR model and the output of this is fed to the LLM. In other words, I think it would be useful to have a cascaded baseline. Although the authors present some qualitative results in the appendix, some quantitative results are necessary where one should use a very strong state-of-the-art ASR model and cascade it with a state-of-the-art LLM since this is a widely adopted pipeline in many speech-to-text chat applications.

2. In line 156, "The modality projector decouples transcription from representation alignment to avoid cross-task interference". Does the speech decoder not get adapted with the alignment? If so, how is it decoupled?
3. The authors propose to use $L_{align}$ as a combination of $L1$ loss and cosine similarity loss such that non-textual and textual features can be decoupled. However, the effect or benefit of this is not clear from table 4. Although we see that non-textual features like emotion and gender information is preserved, we do not know if this can be achieved from just a simple $L1$ loss or cosine-similarity loss.
4. From table 1, it seems that the authors use dialogue data for stage 1 training. How is a non-ASR data used for stage 1, since it only involves the AED ASR model and the modality adapter?
5. In line 231, "for each speech input, the paired text is fed into the LLM backbone to generate the response representing its original behavior." At what stage is this data used?

### Soundness
4

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
This paper introduces AlignChat, a framework to add end-to-end speech understanding capabilities to large language models (LLMs) without finetuning the LLM backbone. 
The key idea is to solve the frequency mismatch between audio and text tokens to enable cross-modal knowledge transfer without requiring finetuning of the LLM. Here, they use an encoder-decoder speech tokenizer based on Whisper features; this architecture autoregressively generates a sequence of speech representations that has a strict one-to-one token-level correspondence with the LLM's text tokens. 
This alignment, refined with a two-stage training process, allows the model to achieve state-of-the-art speech-to-text chat performance using a fraction of the training data of compared methods (about 1/20th).

### Strengths
The results are pretty striking: AlignChat achieves improved performance on OpenAudioBench and VoiceBench using an order of magnitude less speech data than other methods compared (~15K hours). Aligning the speech tokenizer to the LLM representation space without a forward pass of the LLM also reduces computational overhead compared to prior approaches. Components are ablated and also broken out by different tasks within benchmarks to show variance and generalization vs limitations

### Weaknesses
- AlignChat employs an encoder–decoder speech tokenizer initialized with Whisper, but only cites the hours of training data used for AlignChat training for comparison, while most of the compared models are trained from scratch - this should likely be clarified in text for a fair description 
- The model critiques cascaded ASR->LLM pipelines for their latency but introduces its own complex, two-step autoregressive process. Inference now requires running a full autoregressive speech decoder to get the representations, and then running the (frozen) LLM to get the final answer. This could also introduce significant latency, but no inference speed or latency benchmarks are provided.

### Questions
- How much heavy lifting is the pre-trained Whisper encoder doing? I'd love to see a comparison that replaces the frozen Whisper encoder with a non-text-supervised encoder (such as HuBERT) to show how much of the perfromance gains are from the proposed metrci true sources of the performance gains

- What is the real-world inference latency? A key motivation for end-to-end models is speed. It would be crucial to see a comparison of AlignChat's end-to-end latency (time from speech input to final text token) versus a standard ASR->LLM pipeline and an encoder-only model.

- The probing experiments show that non-textual features like emotion and gender are preserved in the speech representations. Could these preserved features be actively used? For example, could the model be prompted to describe the speaker's emotion, or could the LLM's response style be conditioned on the detected emotion, as hinted at in the appendix ?

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
4

### Summary
The paper proposes AlignChat, an E2E speech to text chat framework that enforces one-to-one token-level alignment between speech and text, allowing the model to freeze the LLM backbone and still transfer language ability into the speech task. 

More concretely, a Whisper-initialized encoder–decoder speech tokenizer produces a ~3Hz sequence whose per-token representations are mapped into the LLM’s input embedding space, with the LLM’s embedding layer and output head shared across modalities. Training uses two stages: (1) ASR + alignment without running the LLM’s transformer; (2) instruction tuning with LLM-generated targets while still freezing the LLM. Reported results claim competitive or SOTA speech-to-text chat performance on OpenAudioBench and VoiceBench.

### Strengths
From the point of speech understanding, it's well-motivated. Framing speech to text as a strict per-token alignment problem and then replacing the LLM’s text embeddings in-place with aligned speech representations is interpretable. Sharing the vocabulary/embedding layer further tightens the bridge.

Compare with Flamingo style of approach, it reduced the computation as the input frequency of audio tokens are much lower.

### Weaknesses
My main concern is the encoder-decoder tokenizer 1-1 assumption introduced a performance bottleneck to the proposed approach. LLM has no way to fix any mistake made by the enc-dec. For audio like music, such representation is for sure lossy.

The claim for the model still keep audio information are weak. Gender are very easy to be captured by any encoder. Emotion need to compare with proper baseline. There is no quantitative experiments about if speaker information has been preserved besides t-SNE plot.

### Questions
Does this approach applicable to multilingual?

Do you compared with a simple baseline, using ASR transcription + an pre computed audio embedding?

Do you have error analysis, when the alignment failed or the aligner halluciate?

### Soundness
2

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
The paper introduces AlignChat, a framework that integrates speech understanding into large language models (LLMs) via token-level representation alignment. Unlike prior end-to-end systems that rely on encoder-only speech tokenizers or require LLM fine-tuning (which risks catastrophic forgetting), AlignChat uses an encoder–decoder speech tokenizer designed to achieve precise one-to-one alignment between speech and text tokens.

Key contributions of the paper include:
1. encoder-decoder speech tokenizer: unlike most of the prior work in speech LLMs, this paper proposes to employ an encoder-decoder speech tokenizer, initialized from whisper-larger-v3.
2. two-stage training strategy: the first step aims to achieve exact token-level alignment between speech and text, and the second stage performs end-to-end instruction tuning.
3. Data synthesis: the paper regenerates the text response from available speech datasets to ensure consistent responses to speech and text input.

### Strengths
1. token-level alignment: Introduces an encoder–decoder speech tokenizer that aligns each speech token one-to-one with text embeddings, unlike prior coarse audio-token approaches.
2. No LLM update needed: the LLM remains frozen, avoiding catastrophic forgetting while still enabling end-to-end speech understanding.
3. Data efficiency: Achieves SOTA results on OpenAudioBench and VoiceBench using only ~1/20 data of prior work.

### Weaknesses
I would like the authors to explain how this framework is different from cascading ASR model with an LLM (perhaps plus joint optimization using the LLM prediction loss). In section 3.1.1, it's mentioned:
> For reproducibility, we adopt greedy decoding for both the generation of speech representations and the final text outputs.

I am confused what the speech decoder actually generates during inference and the authors didn't give any more details about it. Hence, the model is built upon whisper and Qwen2.5, in Table 2, there's supposed to be a cascaded system baseline using Whisper + Qwen2.5 (or for more fair comparison, finetune the Whisper on the 15k hour speech datasets) in order to validate the efficacy of the proposed finetuning method. In the performance comparison, it'd be also more informative to compare with some of the commercial multimodal LLMs such as Gemini/GPT4o.

Without the differentiators with cascaded system or comparison with cascaded system, I cannot judge whether AlignChat’s “end-to-end” gains are due to its alignment design or simply comparable to a well-tuned cascaded system. Therefore I have to rate the paper at 2. I am open to discuss more with the authors.

### Questions
find in 'Weakness' section

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This is essentially an end-to-end speech-to-text chat framework that adds speech understanding to a frozen LLM without hurting its core skills. The main essence is token-by-token alignment - a Whisper-based encoder–decoder generates tokens at about the same pace as text, shares the LLM’s tokenizer and embeddings and output head, with a projector layer to map speech representation to the LLM token input space. 

Training methodology

1. ASR style pretraining to align and transcribe without even running the LLM
2. instruction tuning where the LLM itself supplies the targets so that its original text behavior carries over to speech. 

The result/contribution is that it's competitive on OpenAudioBench and VoiceBench, shows tight speech–text alignment and keeps useful non-textual cues (like emotion) while avoiding possible catastrophic forgetting.

### Strengths
The paper combines known methodologies frozen LLMs, Whisper-based speech encoders, and projector-based alignment methods but adds an encoder–decoder speech tokenizer that matches text token rate and shares the LLM's tokenizer/embeddings/output head. I think that this strict alignment is a good step away from the encoder-only audio-token pipelines. The architecture and training flow are also easy and intuitive to follow + the figures are illustrated well. I like to believe that in general this could become a default recipe for adding speech to LLMs without retraining the backbone, and the rate-matching idea may generalize to other modalities in the future..

### Weaknesses
Alignment loss design is heuristic - Report embedding norms and cosine histograms pre/post training; specify whether vectors are normalized before L1/cosine, Ablate α/β, compare to normalized-only (cosine) and contrastive losses (such as for eg InfoNCE with in-batch negatives). token synchronization is the essence of the paper and needs a better metric to justify it. Consider adding something like a token edit distance between inferred token sequence and transcript

### Questions
- Please report full benchmark scores for Stage 1 alone, and the incremental gains of Stage 2 per subset.
- Qwen's tokenizer may segment differently from Whisper's. Did you observe systematic failure modes. Any language/domain-specific mitigations?
- How do you ensure Lspeech ≈ Ltext and maintain 1-to-1 correspondence without teacher forcing? Did you use scheduled sampling/professor forcing or any consistency regularizers? Please quantify alignment drift. How robust is the alignment under fillers, repetitions or partial words? 
- Can you report an explicit "token synchronization error" metric to substantiate the "strict alignment" claim?

### Soundness
3

### Presentation
4

### Contribution
2
