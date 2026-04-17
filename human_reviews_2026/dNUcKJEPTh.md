# Kanade: Compact Linguistically Rich Speech Tokens for Spoken Language Models

- Decision: Reject
- Scores: 4, 2, 4, 4, 2, 6

## Abstract
A good language model starts with a good tokenizer. Tokenization is especially important for speech modeling, which must handle noisy continuous speech recordings. A speech tokenizer should produce compact, linguistically rich representations while still enabling high-quality synthesis. We present Kanade, a tokenizer that realizes this ideal. Kanade separates out acoustic constants like speaker identity from the signal to create a single-stream discrete representation of speech that captures linguistic content, including suprasegmental features. Experiments show that Kanade achieves state-of-the-art speaker disentanglement and linguistic availability while maintaining competitive reconstruction quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors propose a model for tokenizing speech in discrete space. The work aims to create a compact set of tokens for representing speech audio. In particular, the authors try to separate speaker identity by having a global, frame-static encoder and to make the tokens represent linguistic content beyond phonetic. The resulting model shows improved performance on a clean English audiobook, with demonstrations on some additional downstream tasks including ASR, TTS, SID, ASV, and VC. However, the novelty in approach is limited–SSL distillation and using early layers for speaker encoding–which were readily tried in previous studies without core breakthrough in methodology. Moreover, while including diverse downstream tasks (more than usual codec papers), some key caveats that may be induced in their methods remain unclear. These are more elaborated in the Weakness section.

### Strengths
* Better bitrate–intelligibility/quality tradeoff than previous studies in clean English audiobook data (However, this is not fair since, these metrics are ignoring the coding space of global embedding which is not discretized.) 

* Improved performance in downstream tasks with 

* Diverse and interesting analyses with visualizations. Those analyses provide many interesting insights, which are not usually covered in common Codec papers. It has a value as an analysis paper.

### Weaknesses
* Lack of novelty: the main issue of this work is that it severely lacks novelty. The separation of frame-wise and global embedding is dated and included in several previous papers (FACodec, TICodec, LSCodec). 

* The coding complexity of global embedding is entirely ignored. The bitrate is calculated only using frame-wise codes. This omission makes several model comparisons significantly unfair, especially in Figure 4, where the other codecs are using all codes participating in the evaluation. This severely disguises the fact that the actual bitrate of Kanade is infinite to fully reconstruct the audio. The suggested reason for this is that the global embedding “would not be used for language modeling”, which is one niche application of codecs. Putting aside whether language modeling is the primary downstream of codec, if the authors would like to claim so, then the spoken language modeling results in Appendix should be presented as primary, main results. 

* Lack of fairness in model comparison. For the baselines, the authors are too selective in baseline applications by truncating the codebooks. The rationale behind this decision is not well supported. As stated in Section 3.2, they controlled bitrate of the baselines for fairness of comparison. However, the selected bitrates  in Table 1 are not consistently controlled across models, having non-uniform bitrates. Moreover, it is more fair to use all codebooks since the proposed model is using the infinite bitrate of global embedding. Also, since the baselines are not designed for truncated employment of codes in reconstruction tasks, it is more fair to include a full codebook scores along with the truncated ones in Table 1, like in Figure 4.

* Performance improvement might be driven by potential in-domain overfitting. The training and test data distribution are more matching in Kanade than other baseline models since some core metrics in reconstruction, TTS, voice conversion are done in read English speech. Especially, for lower bitrate (putting aside the fact it is broken), could be achieved by not covering diverse speaking styles in the code space. The evaluation on OOD datasets–Expresso, Japanese and L2 speech–and  in Appendix D.6 doesn’t include other baselines. The F0 reconstruction severely drops compared to the in-domain evaluation in Table 1. And some quality metrics are not informative due to the absence of the baselines. If the baselines, for example the full codebooks of Mimi, are better in those OOD data. The reduction in bitrate is likely induced by being limited in the data domain. 

* Lack of support for several key claims. Claim for “linguistically rich.” The conceptual implementation is to merge prosody and phonetic information in a single token stream. However, to claim “richness” in linguistic information, there exists other key aspects like lexical semantics, syntactics, etc. For example, Pasad et al. (20xx, 20xx) show some lexical semantics are represented in speech SSL models. Nonetheless, none of the analyses is supporting evidence for representing those other key linguistic components. ASR and TTS are still operating at the surface level–matching written and spoken formats vice versa. In particular, as shown in Table 13 in Appendix D.4, the proposed tokens are worse in spoken language modeling compared to the previous SSL-distilled Codecs, which imply it lacks those linguistic information. Though the main focus of this paper is targeting the “surface” of the language, which is also denoted well by the authors, being limited in surface level is not compatible with being rich.

### Questions
* For ASR, I wonder if all RVQ codes were used for the baselines. If not, that version should be reported as well for a fair comparison.

* Does the model work well with the speech clips with changing voice quality? For example, if a speaker changes his voicing texture within a sentence, how does the model behave? Does it still reconstruct well for both textures?

* I recommend comparing other similar recent models like TiCodec or LSCodec which have the same branching architecture as Kanade.

* The computing complexity is another useful indication of the utility of codec, especially for real-time downstream tasks. I wonder about the real time factor of Kanade in both encoding and decoding, respectively, compared to the baselines.

### Soundness
3

### Presentation
3

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
The paper proposes Kanade, a two-branch speech tokenizer that takes SSL features, quantizes a content stream with FSQ, and routes speaker- and environment-related constants through a single global embedding for reconstruction. Training uses a joint SSL feature reconstruction and Mel L1 losses, followed by GAN post-training on the decoder. Models are evaluated for reconstruction and for discriminative (ASR/SID/ER) and generative (TTS, VC) downstream tasks. Reported highlights include strong reconstruction intelligibility at low marginal bitrate, competitive TTS, and strong VC disentanglement.

### Strengths
1. Clarity & simplicity of the architecture. A single-stream token sequence, combined with a separate global embedding, is a clean way to keep linguistic content compact and accessible to SLMs while offloading invariants to a non-autoregressive path.

2. Thorough evaluation and ablations. The paper covers reconstruction, discriminative and generative tasks, and includes ablations showing the roles of dual-branch design and loss formulation.

3. Good reconstruction intelligibility and VC disentanglement at low marginal bitrate.

4. Data/computation efficiency claims compared to some prior codecs are appealing.

### Weaknesses
# 1. Limited novelty / unclear source of gains

The global/content disentanglement pattern (local vs. global paths) and using SSL features with VQ-VAE/FSQ are all established. The paper itself situates Kanade within prior disentanglement lines and VQ-VAE tokenizers, emphasizing reduced complexity rather than new principles. A sharper analysis isolating what specifically drives gains (FSQ vs. RVQ, layer choices, GAN post-training) would help.

Some prior works with similar ideas
- https://arxiv.org/abs/2402.08093
- https://arxiv.org/abs/2402.03407
- https://arxiv.org/abs/2309.00169 (RepCodec: a baseline directly related to quantizing speech SSL models, but was not compared here)
- https://arxiv.org/abs/2505.19273

# 2. Evaluation on clean data dominates + OOD robustness

Models are trained on LibriTTS and the main reconstruction is on LibriSpeech test-clean. OOD results (Table 14) show noticeable WER/CER degradation on noisy/emotional/unseen-language sets, sometimes with paradoxically higher UTMOS than ground truth. The authors themselves note sensitivity to dynamic background noise. This limits claims about realistic SLM readiness.

# 3. UTMOS and limited human evaluation

In reconstruction, ground-truth UTMOS is 4.07, while Kanade reaches 4.16 (Table 1), which is counterintuitive and suggestive of predictor bias. Also, the paper includes MUSHRA for reconstruction, but TTS quality is reported only with UTMOS, not human MOS/MUSHRA, despite being a core generative downstream task. This weakens quality claims for TTS.

# 4. Bitrate accounting may overstate efficiency for generative use.
The paper repeatedly uses marginal bitrate, explicitly excluding overheads like Kanade's global embedding. Yet the global embedding is required for resynthesis and is used in downstream setups, so an end-to-end bitrate/conditioning cost (including global) would better reflect the trade-offs. 

# 5. Potential issues with downstream comparisons

- For ASR, RVQ baselines are evaluated with only the first layer (assumed to carry linguistic content). If lexical information is distributed across layers for some models, this setting may disadvantage them. The paper assumes the first layer suffices.

- For TTS, tokenizers without global embeddings are compensated by prepending reference audio streams, while tokenizers with global vectors use extracted globals. These differences can affect fairness and make it hard to attribute gains purely to token quality.

- GAN post-training likely contributes substantially to *quality* metrics. Ablations show notable changes when removing GAN post-training. However, the paper does not thoroughly quantify how much of the perceived quality/improvements versus baselines stem from this step, which other codecs may not share in this exact form.

# 6. Streaming and deployability limits

The authors acknowledge that tokens are not streamable due to a bidirectional SSL backbone; constant-rate tokens may be redundant; robustness to dynamic background noise is limited. These are important for SLMs in the wild.

### Questions
1. **Where exactly do the gains come from?** Please provide controlled ablations that isolate: FSQ vs. RVQ (single-layer RVQ), GAN post-training off vs. on, and precise SSL layer choices for both content and global branches. A compact table summarizing their individual contributions to WER/MUSHRA/UTMOS would help.

2. **Human evaluation for TTS.** Since Table 4 reports only UTMOS, please add a human MOS for TTS to substantiate quality claims and address predictor bias (e.g., your reconstruction UTMOS exceeds ground truth).

3. **Account for global embedding cost.** Provide end-to-end bitrate/conditioning overheads, including the global embedding (e.g., its size, transmission/conditioning cost in a realistic SLM/TTS pipeline) and re-plot rate-distortion curves under this.

4. **Robustness on noisy/emotional/unseen-language speech.** The OOD table reveals notable gaps. Please include targeted experiments with data augmentation/noise-robust training, streamable encoders, or variable-rate tokenization to show mitigation. Also report WER/MUSHRA changes under additive music/babble at different SNRs.

5. **Fairness of downstream baselines.**
    - ASR: Evaluate RVQ baselines with more than one layer (e.g., 1:2) where feasible, to test whether lexical information is indeed concentrated in layer 1 for those models.
    - TTS: Provide an alternative setup where all models use identical prompting/conditioning (e.g., everyone gets a fixed global vector estimated from the same reference frontend) so we can attribute differences to token quality rather than conditioning mechanics.

6. **SSL encoder training.** Kanade uses frozen WavLM features. Please clarify whether any fine-tuning of the SSL encoder was attempted. If not, could light fine-tuning destabilize disentanglement or improve robustness? A small study would answer this open question.

7. **Streaming & latency.** Since non-streamability is a stated limitation, can you prototype a streamable variant (chunked encoding) and report latency/quality trade-offs?

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
3

### Summary
This paper introduces Kanade, which is a speech tokenizer. Kanade contains a content branch and a global branch that are aimed at capturing phonetic features and speaker characteristics, respectively. Trained on just 586 hours of LibriTTS data using WavLM features, Kanade achieves strong performance on downstream tasks, while maintaining competitive reconstruction quality at low bitrates.

### Strengths
The paper addresses a timely and important problem as spoken language models become more prominent. Efficient speech tokenizers will be important for next-generation multimodal language models. The work has clear practical relevance given the growing interest in speech-based LLMs for applications such as voice assistants. The proposed two-branch architecture presents a practical solution that achieves competitive performance across multiple metrics while using only modest amounts of training data.

### Weaknesses
The introduction could benefit from a more direct problem statement. While it establishes the trade-offs current tokenizers face, a clearer explanation of why existing solutions are insufficient before presenting Kanade would strengthen the narrative.

The ablation study for SSL feature layer selection appears non-systematic, as it tests only some layers for content, and fails to comprehensively evaluate layer combinations for the global branch. A systematic sweep would be necessary to validate the chosen layers. The ablation results should be in the main text rather than appendices, as they directly impact core architectural decisions.

The paper frequently references specific experimental results in Section 2 before introducing the experimental setup in Section 3. This forward-referencing forces readers to jump ahead to understand the evidence behind design choices. The method section should either avoid citing specific results or preferably be reorganized for better readability.

Training only on LibriTTS raises questions about generalization. Also, the main evaluation uses mostly read speech datasets, which may not reflect real-world performance on spontaneous speech that spoken language models would need to handle.

Claiming SoTA performance seems not to be substantiated by all presented results, as the scores on many performance metrics appear close between models. Consider adding significance tests or soften claims.

### Questions
Could the authors motivate the choice for WAVLM as the SSL encoder instead of other available SSL encoders?

Could the authors provide more information about the SSL feature layer selection? Specifically, could the authors report a grid/sweep over candidate layers?

The goal of the global branch is to capture speaker characteristics and the authors write that the global branch captures information about the audio that does not change over time. However, many speaker characteristics (pitch, style, emotion) do vary within and across utterances. Could the authors clarify this design decision and contextualize the effect of producing a global embedding only?

To validate that the results generalize beyond read speech, could the authors evaluate their method (e.g., on ASR) from any established corpus containing spontaneous conversational speech? This would strengthen claims about suitability for spoken language modeling, which must handle diverse real-world speech.

In the ASR evaluation, RVQ-based models use only their first layer while FACodec uses both of its content layers. Could the authors clarify this methodological choice and provide additional results with a per layer RVQ ablation?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work present a new speech tokenizer named Kanade designed for speech language models. They use a SSL encoder (WavLM) to extract features and partition the features in different layers into two branches, one for content and the other for speaker. They use a SSL reconstruction and a Mel reconstruction loss to train the model and add a GAN post training to get fine-grained spectrogram. They compare Kanade with many speech codecs, and evaluate generations on WER, quality, speaker identity and prosody. Reconstruction results show that Kanade can achieve comparable or better results with lower bitrate. They also have ASR, speaker and emotion recognition, TTS and VC as downstream tasks to further show the strong ability of Kanade.

### Strengths
1. The experiments are thorough and extensive. Besides the model and speech token itself, this paper presents a comprehensive evaluation protocol of speech tokenizers, including intelligibility, quality, speaker identity, prosody, and some discriminative and generative tasks. Ablation studies and analysis provide clear evidence for the design choices of the speech tokenizer.
2.  The proposed speech tokenizer can achieve high-quality speech generation with just small amount of data and lower bitrate compared to most existing methods.

### Weaknesses
1. In the abstract and introduction of the paper, the authors present the proposed speech tokenizer as an important advance for spoken language modeling results. However, SLMs results are put in appendix and show Kanade underperforms SSL-distilled codecs (SpeechTokenizer, Mimi) on both sWUGGY and sBLIMP, which are important metrics for SLM evaluations. And there are a lot of other SLM evaluation metrics which are not mentioned or considered here. 

2. Missing comparisons with some existing work speech tokenization such as dmel (https://arxiv.org/abs/2407.15835) on basic generative and discriminative tasks. Also, while the author mentioned PAST (https://www.arxiv.org/pdf/2505.14470), Sylber (https://arxiv.org/abs/2410.07168) and SyllableLM (https://arxiv.org/abs/2410.04029) in the related work, there is no comparison with them. They achieve impressive results in SLMs.

3. The proposed speech tokenizer is heavily dependent on some pretrained models, and is a relative straightforward fusion of existing ideas. With the incremental model side innovation, I expect better results in SLMs as the author claimed.

### Questions
1. How do you prove features from different layers contain content and speaker information? Table 3 why using both representations leads to worse performance than using just the global embedding? As the way you partition the features is pretty hard, will there be any leakage of the linguistic information?

2. Can you provide more concrete experiments/evidences why Kanade can lead to better SLMs despite underperforming on sWUGGY/sBLIMP? What model sizes did you train for SLMs? Maybe larger models can lead to more clear gaps?

### Soundness
3

### Presentation
4

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
This paper introduces **Kanade**, a speech tokenizer designed for Spoken Language Models (SLMs). Kanade uses a two-branch architecture that separates linguistic content (phonetic + prosody) from time-invariant acoustic characteristics (speaker identity, recording conditions). The content branch quantizes deep SSL features (WavLM) into discrete tokens at 25Hz using FSQ, while the global branch extracts a single continuous embedding from shallow layers for reconstruction. The paper demonstrates SOTA performance in speaker disentanglement and linguistic richness while maintaining low bitrate (250 bps) and high reconstruction quality.

### Strengths
1. **Well-motivated architecture**: The two-branch design elegantly addresses the fundamental challenge of separating linguistic content from acoustic characteristics. Using deep vs shallow SSL layers is principled and grounded in prior understanding of SSL representations.

2. **Comprehensive evaluation**: The paper includes extensive experiments covering reconstruction quality, speaker disentanglement, linguistic availability (phoneme and prosody), and cross-corpus generalization. The metrics are well-chosen and appropriate.

3. **Compact representation**: Achieving single-stream discrete tokens at 25Hz is a significant advantage for downstream LM efficiency compared to multi-stream RVQ codecs.

### Weaknesses
1. **Insufficient SLM experiments**: This is the most critical weakness. The paper is explicitly motivated for "Spoken Language Models" and positions Kanade as a tokenizer designed FOR SLMs, yet provides minimal experimental validation for actual language modeling. The main paper includes no SLM experiments at all. The only language modeling results appear in Appendix Table 13, which shows preliminary next-token prediction experiments on a small LibriSpeech dataset. Critically, these results demonstrate significantly worse performance compared to existing published speech LMs such as Moshi and Kimi-Audio, raising serious concerns about whether Kanade tokens are actually suitable for the proposed use case. Without convincing evidence that these tokens work well for autoregressive language modeling, the paper's core contribution remains unvalidated.

2. **Questionable SOTA claims and insufficient experimental details in Table 4**: The TTS experiments in Table 4 lack transparency and fairness, making it difficult to assess the validity of the reported results:
	1. **Missing training details**: The paper provides no information about what data were used to train the models in Table 4, nor details about model architecture, parameter counts, or training procedures.
	2. **Unfair comparison across tokenizers**: Different speech tokens are optimized for different modeling architectures. Using a single architecture optimized for Kanade to evaluate all baselines creates an unfair comparison. For example, Mimi uses a depth transformer specifically designed for RVQ decoding—was this architecture used in Table 4, or were all baselines forced into Kanade's decoder?
	3. **Lack of baseline context**: Table 4 includes no comparison with published or publicly released TTS models. Without established benchmarks, it is difficult to interpret whether 4.7% WER represents strong performance, especially given that the test set appears to be newly created by this submission rather than using widely adopted benchmarks like seed-tts-eval (https://github.com/BytedanceSpeech/seed-tts-eval).

3. **Missing critical related work and insufficient contextualization**: The paper fails to cite and compare with several highly relevant recent works, and tables lack proper contextualization with SOTA baselines:
	1. **SparkTTS** (https://arxiv.org/pdf/2503.01710): Introduces a new speech tokenizer and demonstrates how to adapt a text-only LLM with speech tokens to enable speech generation—directly relevant to the SLM motivation.
	2. **dMel** (https://arxiv.org/abs/2407.15835): Proposes a tokenizer-free discrete speech representation that can be jointly modeled with text tokens in transformer-decoder architectures for both TTS and ASR.
4. **Insufficient baseline contextualization**: All downstream evaluation tables (Tables 2, 4, 5) omit comparisons with SOTA task-specific models. For instance:
		- Table 2 should report the latest SOTA ASR WER, speaker verification results, and emotion recognition accuracy to help readers understand where Kanade's performance stands relative to specialized models (not necessarily to claim superiority, but to provide context).
		- Table 4 should include results from existing TTS models for reference.
		- Table 5 should include voice conversion results from open-source models to establish performance context. 


Overall, my concern is about the experiments:
- use the submission's own test set and their own model, without using standard benchmark and opensourced models
- no experiments about SLM in the main body. Only find minimal exploration in appendix. This is not aligned with the abstract and the introduction of this submission

### Questions
- How robust the proposed speech token is? For example, can it reconstructed speech with noise background as described in dMel paper(https://arxiv.org/abs/2407.15835)? 

- Can you evaluate the TTS model with seed-tts-eval benchmark (https://github.com/BytedanceSpeech/seed-tts-eval) or Emergent-TTS-Benchmark (https://arxiv.org/abs/2505.23009)? otherwise hard to understand the results with your own test set.

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
The paper presents Kanade, a speech tokenizer which aims to produce compact, linguistically rich, and speaker-disentangled speech representations. It achieves this through a dual-branch encoder comprising a content branch which captures linguistic and prosodic information and a global branch which captures suprasegmental features such as speaker style and channel characteristics. The content branch would be responsible for tokenizing in the manner similar to a text encoder, while the global branch would separate the speech-specific context. Content branch employs a zero-mean, unit-variance normalization, a transformer encoder with local-window attention and strided convolution for downsampling followed by quantization. Global branch uses ECAPA-TDNN with ConvNeXt and then Adaptive stats pooling for a single global vector. The model is trained with a joint mel-spectrogram reconstruction loss and an SSL feature reconstruction loss to balance prosody, acoustic detail, and linguistic fidelity. The paper additionally employs some design choices that strengthen the work, including Finite Scalar Quantization (FSQ) for quantization to prevent collapse, conditioning the mel spectrogram generation on the global embedding via AdaLN-Zero, and GAN post training to ensure the reconstructed speech quality is unaffected. 

The results show positive performance on both generative and discriminative speech tasks. The paper also includes an extensive suite of ablations in the supplementary section. Overall, the paper presents a clean architecture combining dual-branch encoding, scalar quantization, and joint Mel+SSL objectives into a stable and compact tokenizer suitable for spoken-language modeling.

### Strengths
* The paper presents a clean separation between linguistic (content) and acoustic/speaker (global) factors. This design makes both empirical and conceptual sense, as demonstrated by the experimental results.
* The identification of Finite Scalar Quantization (FSQ) as an efficient and stable compression alternative is a notable contribution. FSQ avoids codebook collapse while achieving extremely low bitrates and maintaining high reconstruction quality. This is arguably the paper’s strongest technical contribution. 
* The inclusion of a GAN-based post-training phase enhances perceptual quality. While not novel, it demonstrates a high level of engineering completeness.
* The paper employs creative and well-designed visualizations to illustrate evaluation metrics (Figures 1 and 4), effectively supporting its empirical claims.
* Paper includes an extensive supplementary section include extensive ablations, design rationales, and component analyses, reflecting thoroughness and careful empirical validation.

### Weaknesses
* The dual-branch architecture (content + global) is claimed to be the first and hence major contribution. However, this approach is not new - prior works such as FACodec (Chen et al., 2023) already separate linguistic and speaker/prosodic factors as content and global branches.
* The paper does not clearly describe whether the content branch alone functions as the tokenizer at inference time. This obscures the model’s practical usage and efficiency trade-offs.
* The value for loss balancing coefficient α between the mel-spectrogram and SSL feature losses is not reported. Since α directly governs the trade-off between prosody and linguistic fidelity, including a sensitivity study on α would benefit the paper.
* The paper suffers from over-reliance on supplementary material. Most of the essential information such as related work and baseline definition is deferred to the supplementary when they should appear in the main paper.
* The paper would benefit from examples or discussions of failure cases or degradation patterns, providing insight into when and why Kanade struggles (e.g., prosodically rich or noisy speech).

### Questions
Questions:
1. The paper mentions “superficial”  features but does not define what counts as “superficial.” Could you clarify this? 
2. During inference, is the content branch alone used as the tokenizer? What does the full tokenization inference loop look like?

Suggestions:
1. The work would be significantly strengthened by explicitly juxtaposing its design and motivations with earlier dual-branch and global-conditioning frameworks, clarifying what is architecturally or functionally distinct about Kanade's formulation.
2. Figure (2) shows other contemporary and prior approaches. The figure makes it seem like Kanade is the only approach with a global branching. Please consider expanding the image to include prior works with similar branching so you can highlight key differences with your method.

### Soundness
3

### Presentation
2

### Contribution
3
