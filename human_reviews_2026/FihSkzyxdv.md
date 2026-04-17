# VibeVoice: Expressive Podcast Generation with Next-Token Diffusion

- Decision: Accept (Oral)
- Scores: 2, 8, 8, 8, 8, 6

## Abstract
Generating long-form, multi-speaker conversational audio like podcasts poses significant challenges for traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking. We present VibeVoice , a novel model designed to synthesize expressive, long-form speech with multiple speakers in a zero-shot manner. A core component of our approach is the continuous speech tokenizers operating at an ultra-low frame rate of 7.5. This tokenizer effectively preserves audio fidelity while significantly boosting computational efficiency for processing long sequences. To facilitate training on authentic conversational dynamics, we have developed an annotation pipeline that generates pseudo transcriptions and turn-taking labels for extensive podcast data. Leveraging this data and our efficient tokenizer, VibeVoice  employs the next-token diffusion framework. This enables VibeVoice  to: (1) synthesize long-form speech (up to 30 minutes) with up to 4 speakers, surpassing the typical 1-2 speaker limits of many prior models; and (2) achieve a high degree of naturalness in turn-taking, pacing, and the rendition of subtle non-lexical cues (such as breaths and lip smacks), which are crucial for listener immersion and capturing the authentic vibe of expressive conversations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces VibeVoice, a novel zero-shot Text-to-Speech (TTS) model designed for synthesizing long-form, multi-speaker conversational audio (e.g., podcasts, up to 30 minutes with up to 4 speakers). This addresses the limits of traditional TTS in handling scalability, speaker consistency, and natural turn-taking.

A core technical element is an ultra-low frame rate (7.5 Hz) continuous speech tokenizer that significantly boosts computational efficiency for long sequences while preserving audio fidelity. Using an extensive new dataset with pseudo transcriptions and turn-taking labels, VibeVoice employs a next-token diffusion framework. This enables the model to achieve a high degree of naturalness in turn-taking, pacing, and subtle non-lexical cue rendition (like breaths), crucial for authentic, immersive conversational output.

### Strengths
The paper addresses an important problem in speech generation. Synthesizing long-form, multi-speaker conversational audio is highly relevant for real-world applications such as podcast generation, enhanced virtual assistants, and dialogue systems. The scenario is undoubtedly innovative and highly applicable

### Weaknesses
The core weaknesses of this work lie in a lack of sufficient technical novelty and inconsistent experimental evaluation.

1. Methodology: The next-token diffusion (or next-distribution) framework, a central component of VibeVoice, has been previously introduced in the literature [1]. The paper does not clearly articulate the incremental technical novelty VibeVoice contributes beyond adapting this existing framework.

2. Scenario: The goal of synthesizing speech for multi-speaker (more than two people) scenarios, particularly for long conversational contexts, is also not entirely new. Prior work [3,4], has specifically designed and discussed methods for long conversational speech generation for podcasts and similar applications for more than 2 people [2].

3. While the paper addresses long-context dialogue generation, it fails to include a fair and direct comparison against key existing models that tackle similar challenges in multi-speaker and long-form synthesis [3,4]. This makes it difficult to ascertain the true performance gain of VibeVoice.

4. The 7.5Hz tokenizer appears to be the core technical contribution, yet no sufficient details are provided to fully evaluate its novelty or effectiveness

5. There is very little detail about the data preparation pipeline.

Moreover, there are many inconsistent and incomplete experimental evaluation:

1. Baseline Inconsistency: The choice of baselines across the evaluation tables is inconsistent. The paper should use a uniform set of relevant baselines across all comparative tables (e.g., Table 1 and Table 2) to provide a clear and comprehensive performance picture.

2. Missing Subjective Metrics: Table 1, which appears to focus on multi-speaker/turn-taking quality, lacks subjective (human) evaluation metrics for some systems. 

3. Missing Single-Speaker Results: The evaluation in Table 2 is incomplete as it lacks results for the single-speaker scenario. 

Finally, the paper fails to include a complete and up-to-date list of references for all cited or discussed works. This is an essential requirement for a conference paper.

[1] Zhu, Xinfa, Wenjie Tian, and Lei Xie. "Autoregressive speech synthesis with next-distribution prediction." arXiv preprint arXiv:2412.16846 (2024).

[2] Xie, Kun, et al. "Fireredtts-2: Towards long conversational speech generation for podcast and chatbot." arXiv preprint arXiv:2509.02020 (2025).

[3]Zeghidour, Neil, et al. "Streaming sequence-to-sequence learning with delayed streams modeling." arXiv preprint arXiv:2509.08753 (2025).

[4] Ju, Zeqian, et al. "MoonCast: High-quality zero-shot podcast generation." arXiv preprint arXiv:2503.14345 (2025).

[5] Zhang, Leying, et al. "CoVoMix: Advancing zero-shot speech generation for human-like multi-talker conversations." Advances in Neural Information Processing Systems 37 (2024): 100291-100317.

[6] Mitsui, Kentaro, Yukiya Hono, and Kei Sawada. "Towards human-like spoken dialogue generation between ai agents from written dialogue." arXiv preprint arXiv:2310.01088 (2023).

### Questions
1. VibeVoice claims to synthesize long-form audio with up to four speakers. Given the high risk of speaker timbre drift and identity confusion in long-context, zero-shot TTS, what explicit architectural mechanisms are implemented to ensure robust and consistent speaker similarity for all four voices?

2. How to prepare and collect data for more than two speakers? For transcription, it is unclear whether to rely only on pseudo-transcription, or should real data also be used? Furthermore, why doesn't pseudo-transcription negatively impact model performance?

### Soundness
2

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
4

### Summary
VibeVoice is a next-token diffusion multispeaker long-context TTS model that benefit from the following innovations: 1) a 7.5 Hz framerate tokenizer; 2) an annotation pipeline that generates pseudo transcription and turn-taking labels.

### Strengths
1. this work studies an important problem - long-form conversation generation, and is one of the few papers in this field. The idea of using continuous features leads to significantly reduced sequence length which enables long context.
2. the data annotation pipeline is an important contribution

### Weaknesses
1. a big contribution of the the paper as noted in the abstract is data pipeline, but it's described in appendix, and there is not evaluation on the design choices of the data pipeline
2. the writing could be improved, for example, it should be noted that in the prompt part, the speech features are just short segments, while the text are the entire conversation (including both prompt and what's to be generated?)
3. missing RTF

### Questions
1. why do we use a $\sigma\text{-VAE}$ as opposed to just an Autoencoder for acoustic feature?
2. In appendix G, subjective eval interface is annotated with Chinese, are human listeners in the subjective evaluation native speakers of English?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents **VibeVoice**, a system designed for **long-form, multi-speaker conversational text-to-speech (TTS)** synthesis. The work addresses three major challenges in the field, **scalability**, **speaker consistency**, and **natural turn-taking with long-context stability**, particularly in the context of podcasts and other extended dialogues.

VibeVoice adopts a **hybrid acoustic–semantic representation**, where speech and text are encoded separately and later fused. The semantic stream is modeled using a fine-tuned **Qwen2.5 language model**, while the acoustic stream is generated via a **next-token diffusion model** that predicts continuous latent acoustic features at **7.5 Hz**.

In addition, the authors introduce a **customized data annotation pipeline** that includes segmentation, speaker diarization, and automatic filtering of low-quality samples.

Empirical results demonstrate improved **turn pacing**, **context-aware prosody**, and **speech naturalness** over long durations (up to 45 minutes), outperforming several leading commercial and open-source speech generation systems in both **subjective and objective evaluations**.

### Strengths
1. **Comprehensive System for Long-Form Conversational TTS**  
- The paper presents a well-integrated end-to-end system capable of generating long, multi-speaker conversations with smooth and natural conversational flow, a significant advancement for practical TTS applications.

2. **Innovative Representation and Generation Mechanism**  
- The use of a **7.5 Hz hybrid acoustic–semantic representation** effectively reduces context length for speech modeling.
- The **autoregressive next-token diffusion** architecture is a sound and well-motivated design choice for achieving high-quality, coherent speech synthesis.

3. **Robust Data Preparation Pipeline**  
- The proposed data pipeline, combining segmentation, diarization, and transcript filtering, demonstrates careful system design and contributes to improving training data quality.

4. **Thorough and Convincing Evaluation**
- The model is extensively evaluated through both subjective and objective metrics.  
- The **subjective evaluation includes 6 hours of audio** and **over 24 evaluators**, which makes the results convincing for an academic paper.  
- The **blind subjective evaluation setup (Appendix G)** appears well-designed and appropriate for assessing perceptual quality.

### Weaknesses
Overall, I did not find any obvious weaknesses. There are minor typographical issues. For example, in **Table 3**, the last row appears to incorrectly include an *Nq* value even though the proposed speech tokenizer does not use a quantizer.

### Questions
1. Could the authors clarify the **key differences between VibeVoice and MoonCast**?  
2. In the demo, the model appears capable of **singing**. Is this behavior intentional or emergent? Additionally, the singing quality seems suboptimal, are there potential ways to improve this aspect?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a method for generating long-form, multi-speaker audio podcasts. It introduces a continuous hybrid (semantic and acoustic) speech tokenizer that achieves high-quality audio reconstruction at a frame rate of 7.5 Hz. Building upon this, VibeVoice employs next-token diffusion to enable high-quality audio generation.

### Strengths
The authors design a continuous hybrid (semantic and acoustic) speech tokenizer that achieves high-quality audio reconstruction at 7.5 Hz.  
They leverage next-token diffusion to enable high-quality audio generation.  
An efficient pipeline for processing raw podcast data is established.  
The method achieves impressive results compared to top-tier models.

### Weaknesses
The proposed method in this paper draws heavily on the design of LatentLLM. Moreover, the designs of both the semantic tokenizer and the acoustic tokenizer have already been extensively explored in numerous audio-related works.

### Questions
Why does the acoustic tokenizer perform well with a single speaker, but exhibit a significant drop in WER when a second speaker is introduced? Could this be due to insufficient training on multi-speaker data?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new model VibeVoice  for long-form zero-shot text-to-speech. 
VibeVoice can generate up to 90 minutes of speech from a text script and a voice-condition, supporting up to 4 speakers and outperforms SOTA models on podcast generation (as shown by quantitative and qualitative metrics).

This method relies on key design choices:
- a separation of acoustic and semantic tokens to get the best of both worlds between text intelligence and acoustic reconstruction
- ultra low frequency (7.5 Hz) tokens (this speeds up LLM inference time)
- next-token-diffusion objective (continuous targets) as opposed to the standard cross-entropy objective (discrete targets).

The authors show extensive ablatation experiments to validate their design choices and compare their generations to state of the art TTS models. They provide detailed training details, code and sample outputs (long-form generated conversations) for reproducibility.

### Strengths
- SOTA model quality on long-form podcast generation
- Great level of detail on all fronts : tokenizer training, LLM training, data preprocessing, inference settings
- original contribution of concatenating acoustic and semantic tokens for LLM input - with a soft target (diffusion). They do ablations on those design choices to validate them
- extensive ablation studies on design choices (tokenizer choice, base model size, diffusion inference settings)
- extensive evaluation on the model, with a new benchmark VIBEVOICE-Eval  + subjective human evaluation

### Weaknesses
- The WER metric on the CFG figure is not very stable, there are no clear trends (as opposed to SIM-O which shows clear trends). This questions the robustness of WER metric in the comparison figures to other models.
- This paper compares to other models in terms of performance but not speed, the  inference time appendix does not show other models as comparison

### Questions
clarifications:
- what is the diffusion training schedule and loss weighting that was used? Same question regarding inference schedule
- what number of diffusion steps was used for all metrics reported in comparison table?
- on the tokenizer ablation , it looks like using hybrid only shows benefits at 2+ speakers, while acoustic is better at 1 speaker. Could you elaborate on this?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents VibeVoice, a framework for zero-shot, expressive, long-form, multi-speaker podcast generation. It introduces two ultra-low frame rate latent (acoustic and semantic) operating at 7.5 Hz and integrates them into a large language model with a next-token diffusion mechanism for scalable, high-fidelity audio synthesis. The system is capable of generating podcasts up to 90 minutes with up to four speakers.

### Strengths
1. The paper tackles an important and challenging task—long-form podcast generation, which combines the difficulties of expressive multi-speaker speech synthesis, discourse coherence, and long-context modeling. Addressing such a complex problem is both timely and valuable for advancing speech generation research.
2. The proposed approach is methodologically reasonable and well-motivated. The reduction of frame rate to 7.5 Hz effectively shortens the sequence length for long-context modeling. The design choices—including (1) the use of continuous latent representations rather than discrete tokens, (2) the hybrid modeling of acoustic and semantic features, and (3) the introduction of a diffusion head to improve latent prediction—all contribute to handling the challenges of long-form expressive audio generation.
3. The evaluation includes a variety of subjective metrics.

### Weaknesses
1. Unclear Training Data Source and Scale: The training data is vaguely described as an internal pseudo-labeled podcast collection without specifying its source, composition, or duration. While the paper mentions ~80 billion training tokens, this could roughly correspond to around 3 million hours of audio, which is an enormous scale. Clarifying the dataset origin and scale is critical for assessing the reproducibility, fairness, and ethical validity of the work.

2. Lack of Analysis on Frame Rate (7.5 Hz) Choice. The paper repeatedly emphasizes the importance of the 7.5 Hz frame rate but does not justify why this particular rate is optimal. It would be highly informative to include experiments at alternative frame rates (e.g., 12.5 Hz or 15 Hz) to show how longer sequences affect LLM modeling difficulty, as well as lower rates (e.g., 3.75 Hz) to measure the trade-off between sequence compression and degradation in WER, expressiveness, or audio quality.

3. Limited Discussion on Scaling and Comparability. Since most compared baselines (e.g., ElevenLabs, Gemini) are black boxes with unknown data and architectures, the fairness of comparison is somewhat constrained. It would be helpful if the authors could supplement scaling experiments, such as: (1) Varying the amount of training data to show scaling behavior. (2) Increasing model capacity (e.g., from 7 B to 30 B) to explore potential performance gains. Such results would make the conclusions more robust and generalizable.

### Questions
1. I wonder to know that should we feed the text scripts at the beginning of sequence all at once (like VibeVoice), or in an interleaved manner (e.g., alternating segments of text and audio, as in FireRedTTS2)? Which one is better?

2. The paper frequently refers to “tokenizers,” but the representations used are actually continuous latent variables, not discrete tokens. It might be clearer to use terminology such as latent encoders or continuous representation extractors to avoid confusion.

3. In Figure 3(b), the SIM-o metric decreases as the number of diffusion steps increases. Why does this happen? Intuitively, more denoising steps might be expected to yield smoother or more consistent outputs—so the observed negative correlation deserves explanation.

### Soundness
3

### Presentation
3

### Contribution
3
