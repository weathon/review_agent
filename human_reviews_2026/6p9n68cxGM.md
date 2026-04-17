# EmoSteer-TTS: Fine-Grained and Training-Free Emotion-Controllable Text-to-Speech via Activation Steering

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
Text-to-speech (TTS) has shown great progress in recent years. However, most existing TTS systems offer only coarse and rigid emotion control, typically via discrete emotion labels or a carefully crafted and detailed emotional text prompt, making fine-grained emotion manipulation either inaccessible or unstable. These models also require extensive, high-quality datasets for training. To address these limitations, we propose **EmoSteer-TTS**, a novel **training-free** approach, to achieve **fine-grained** speech emotion control (conversion, interpolation, erasure) by **activation steering**. We first empirically observe that modifying a subset of the internal activations within a flow matching-based TTS model can effectively alter the emotional tone of synthesized speech. Building on this insight, we then develop a training-free and efficient algorithm, including activation extraction, emotional token searching, and inference-time steering, which can be seamlessly integrated into a wide range of pretrained models (e.g., F5-TTS, CosyVoice2, and E2-TTS). In addition, to derive effective steering vectors, we construct a curated emotional speech dataset with diverse speakers. Extensive experiments demonstrate that EmoSteer-TTS enables fine-grained, interpretable, and continuous control over speech emotion, outperforming the state-of-the-art (SOTA). To the best of our knowledge, this is the first method that achieves training-free and continuous fine-grained emotion control in TTS. Demo samples are available at https://emosteer-tts-demo.pages.dev/.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes EmoSteer-TTS, a novel framework for fine-grained, continuous emotion control in Text-to-Speech (TTS) models. The key innovation is that this method is training-free; it can be applied to existing, pretrained flow matching-based TTS models (e.g., F5-TTS, CosyVoice2) without any fine-tuning.

The method works by "activation steering." First, the authors compute "steering vectors" by finding the average difference in internal model activations between neutral and emotional speech, using a curated dataset. This vector is then refined by identifying the "top-k" most emotionally salient activation tokens, as validated by an external Speech Emotion Recognition (SER) model. At inference time, this steering vector is added to the model's activations with an adjustable strength parameter (α), allowing for precise control.

### Strengths
The core contribution, a training-free, "plug-in" framework for fine-grained emotion control, is novel for EC-TTS. It obviates the need for costly retraining or large, multi-emotion datasets for each new TTS model.

The paper validates EmoSteer-TTS on three different SOTA flow-matching models, demonstrating its general applicability within this model class. The evaluation is robust, combining objective and subjective metrics, and includes an OOD test.

### Weaknesses
"Training-Free" vs. "Data-Dependent": The "training-free" claim is slightly misleading. While the TTS model is not trained, the method requires a non-trivial offline process: (1) curating a substantial (6,900-sample) high-quality emotional speech dataset, and (2) using a separate, pretrained SER model (emotion2vec) to process activations and find the top-k tokens. The success of the method is therefore highly dependent on the quality of this curated dataset and the accuracy of the chosen SER model. The sensitivity to these components is not explored.

The authors admit in the limitations (Line 511) and in the text (Line 411) that strong steering (α) can introduce artifacts and unintelligible speech. This is a critical and expected limitation. However, the paper is missing a crucial experiment that quantifies this trade-off. An ablation study plotting steering strength (α) against audio quality (N-MOS) and intelligibility (WER) would be necessary to understand the practical usable range of emotion control.

The method is exclusively demonstrated on flow matching-based TTS models using a DiT backbone. It is an open question whether the core assumption—that emotion is represented in a linearly steerable subspace of activations—generalizes to other dominant TTS architectures, such as VITS (VAE/GAN-based) or codec-based autoregressive models (e.g., VALL-E). This limits the generality of the paper's claims.

### Questions
The top-k token selection (Sec 3.3) relies on emotion2vec. How sensitive is the quality of the resulting steering vector to this choice? For instance, what happens if SenseVoice (which was used in the evaluation) is used to generate the steering vectors instead? Does performance degrade?

The claim of composite emotion via linear addition (Eq. 11) is very interesting. Could the authors comment on the qualitative results? Does "anger + sadness" sound like a convincing blend, or does it sound muddled/confused?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a training-free method for achieving continuous fine-grained emotion control in TTS synthesis. While the implementation details are mostly provided, several aspects of novelty, clarity, and reproducibility remain unclear. I appreciate the potential contribution toward controllable emotional TTS but there are significant concerns regarding the method’s originality, clarity of presentation, and empirical validation.

### Strengths
The idea of training-free fine-grained emotional control is interesting for advancing expressive TTS systems. If it can well explained and validated, the proposed approach has the potential to reduce the reliance on large paired emotional datasets, which remains a challenge in emotional TTS.

### Weaknesses
This paper can be further improved by addressing the limitations including insufficient literature coverage, unclear method design, limited novelties, weal results and unclear reproducibility details.

The methodology section is not clearly written. I suggest improving it by explaining the underlying motivation and the rationale behind the design choices. Additionally, please clarify what each equation represents and how it contributes to the overall approach.

The related work section would benefit from including key studies on label-based EC-TTS approaches, as their omission currently makes the authors’ claim less convincing.

I am not fully convinced by the design of the proposed approach, and the paper appears to lack sufficient novelty for ICLR.

The synthesized samples do not clearly reflect effective emotion control, particularly for emotions such as disgust, happiness, sadness, and surprise.

### Questions
Do the authors plan to release the curated emotional speech dataset along with the data processing procedures for public use? Additionally, please clarify which testing data and how many utterances were used in the experiments. Would it be possible to evaluate the proposed approach on each dataset separately, and how might this affect its performance?

While the implementation details are mostly described, I wonder whether the authors could release the code to confirm the mathematical details and facilitate reproducibility. If possible, could the authors make it available on the demo page? If not, please provide an explanation.

Why do you think that your approach is the first method that achieves training-free and continuous fine-grained emotion control in TTS? There are several zero-shot TTS models that appear to offer similar capabilities. Please review the related works and provide a fair, detailed comparison against these approaches.

Do the authors plan to release the curated emotional speech dataset along with the data processing procedures for public use? Additionally, please clarify which testing data and how many utterances were used in the experiments. Would it be possible to evaluate the proposed approach on each dataset separately, and how might this affect its performance?
The motivation could be further strengthened by referencing and discussing more recent works on emotional TTS.

What are your novelties for activation steering compared with those in text-to-image models?

The authors claim that label-based approaches rely on paired emotional data, suggesting that their method eliminates this dependency. However, the proposed approach still requires calculating activation differences using paired samples. What distinguishes this method from existing label-based approaches? Moreover, several prior works already achieve comparable results using only a small amount of emotional speech data for training.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes EmoSteer-TTS, a training-free method for fine-grained emotion control in text-to-speech via activation steering. The key idea is to identify and manipulate a sparse subset of internal activations in flow matching, DiT-based TTS models (F5-TTS, E2-TTS, CosyVoice2) to modulate emotional tone without additional training. The method constructs per-emotion steering vectors by computing difference-in-means between activations elicited by neutral vs. emotional reference speech, then selecting top-k emotion-relevant token positions using a SER probe to form a weighted steering vector. At inference, the model steers activations with strength α for conversion/interpolation, erases target emotions with strength β via projection-based subtraction, and supports composite operations (replacement or multi-emotion blending). The approach is plug-and-play, adding lightweight hooks to the first residual stream of selected layers and CFM steps.

Empirically, EmoSteer-TTS provides continuous and interpretable control over emotion intensity, outperforming or matching strong baselines on emotion similarity while maintaining intelligibility and speaker similarity, and shows smooth interpolation and effective erasure in both in-distribution and OOD settings. The paper contributes a curated emotion dataset (6,900 utterances) for steering vector construction, detailed implementation hooks, and a thorough analysis of “emotion steering dynamics” across k, layers, and steps. Limitations include reliance on curated emotional references to build steering vectors, potential circularity from using emotion2vec for both probing and evaluation, incomplete fairness in baseline comparisons, limited prosodic analyses beyond F0, artifacts at higher α, and missing efficiency measurements. Overall, the work introduces a novel, practical, and interpretable training-free control mechanism for EC-TTS, with compelling results and clear paths for strengthening evaluation and analysis.

### Strengths
* Originality

  - Introduces a training-free, activation-steering paradigm for emotion control in TTS, a clear departure from the prevailing label- or description-conditioned methods that require large-scale training and supervision.
  - Creatively adapts activation steering—previously shown effective in LLMs and T2I diffusion—to flow-matching, DiT-based TTS models, demonstrating cross-domain transfer of a control technique to speech generation.
  - Proposes a principled pipeline to discover emotion-relevant internal tokens: difference-in-means activation extraction between neutral/emotional references, top-k token selection via SER-driven probing, and weighted steering vectors. This yields interpretable, fine-grained control at inference.
  - Expands the control space beyond discrete labels and text prompts to continuous strengths, interpolation between emotional states, erasure of target emotions, and composite manipulation (replacement and multi-emotion blending).
  - Offers analysis of “emotion steering dynamics” across layers, steps, and top-k choices, giving novel insight into how emotion is encoded and can be modulated within flow-matching TTS architectures.

* Quality

  - Extensive empirical evaluation across three strong, diverse backbones (F5-TTS, E2-TTS, CosyVoice2), showing the method is plug-and-play and broadly applicable.
  - Both in-distribution and out-of-distribution tests are reported, with robust performance and minimal degradation, strengthening claims about generalization.
  - Uses multiple complementary metrics: intelligibility (WER), speaker preservation (S-SIM), emotion similarity via two different SER models (emotion2vec and SenseVoice) to reduce metric overfitting risks, and listener studies (N-MOS, EI-MOS, EE-MOS) for perceptual validation.
  - Demonstrates fine-grained control via smooth interpolation curves and F0 contour visualizations; shows effective erasure and composite control with clear quantitative and qualitative evidence.
  - Provides ablations and analyses on top-k, steered layers, and CFM steps, which clarify design choices and the method’s operating regime.
  - Releases detailed implementation hooks and reproducibility details; constructs and describes a curated emotional speech dataset to derive robust steering vectors.

* Clarity

  - The paper is well-structured with clear motivation (limitations of label/prompt-based EC-TTS), methodological overview figures, and concise mathematical formulations of activation extraction, steering, interpolation, erasure, and composite control.
  - Figure 3 and the step-wise algorithm description make the steering pipeline easy to follow; equations use intuitive normalization and projection operations explained in context.
  - Appendices provide code snippets, metric definitions, model configurations, dataset curation, and additional visualizations—substantially improving reproducibility and reader understanding.
  - The limitations are candidly stated (dependency on high-quality emotional samples for steering vector construction, potential artifacts at high steering strengths), which helps situate the claims.

* Significance

  - Addresses a long-standing pain point in EC-TTS—coarse, unstable, and training-heavy control—by enabling stable, continuous, fine-grained emotion manipulation without additional training or large labeled datasets.
  - The training-free, inference-time approach lowers the barrier to adoption: practitioners can retrofit existing TTS models to gain controllability without re-training or data collection, which is impactful for applied scenarios.
  - Composite control (replacement and multi-emotion blending) broadens the expressive repertoire beyond single-label styles, enabling richer user experiences and fine editorial control in applications such as storytelling, assistive agents, and audio post-production.
  - The interpretability of steering vectors and token-level masks provides a new lens to study how emotional tone emerges in TTS models, potentially influencing future architecture designs and control strategies.
  - As the first method (to the authors’ knowledge) achieving training-free, continuous fine-grained emotion control in TTS, EmoSteer-TTS is likely to spur follow-up work at the intersection of controllability, interpretability, and efficiency in speech generation.

### Weaknesses
The approach, while training-free at inference, still relies on a curated pool of high-quality emotional speech to build steering vectors, which weakens the claim of being data-free and raises questions about scalability. Please quantify sample complexity (how many and what quality of references are needed), test cross-lingual transfer (build in one language, apply to another), and assess robustness to noise, reverberation, and device/domain mismatch.

Token selection and several evaluations depend on emotion2vec, creating potential circularity and model bias. Beyond adding SenseVoice for evaluation, diversify both probing and evaluation: use multiple heterogeneous SERs (including VAD regressors), include human emotion identification/AB tests with confusion matrices, and explore alternative token-attribution methods (e.g., linear probes, integrated gradients, activation patching, RSA) to ensure the discovered tokens are not artifacts of a single SER.

Comparisons against baselines are weakened by reliance on demo samples and “unguaranteed” reproductions. Re-run competitive training-based and prompt-based EC-TTS under controlled protocols (same text/reference, recommended hyperparameters), and report perceptual results with full details: rater counts, inter-rater reliability, confidence intervals, and significance tests. This is important to substantiate the “first training-free fine-grained control” claim.

The interpretability story—that a sparse subset of tokens encodes emotion—remains preliminary. Provide stronger causal evidence by patching/swapping activations across time/layers, disentangling from phonetic content and speaker traits, and mapping top-k indices to prosodic correlates and VAD dimensions. Time-localization analyses showing whether selected tokens align with prosodic modulation regions would make the findings more convincing.

At higher steering strengths, artifacts appear, and the control lacks principled safeguards. Consider adaptive strength schedules across layers/steps, subspace-constrained steering, or calibration that sets α based on projection magnitude onto the steering vector. Report intelligibility/naturalness as continuous functions of α per backbone to give users safe operating ranges.

Prosodic analysis focuses mainly on F0; this underrepresents emotional variation. Include energy contours, duration/speaking rate, pause statistics, spectral tilt, jitter/shimmer, or eGeMAPS features, and show that erasure drives these toward neutral baselines while interpolation varies them smoothly with α.

Finally, efficiency and composite control need deeper treatment. Quantify runtime/latency overhead (RTF, memory) for different k, layer/step settings, and streaming scenarios, and provide ablations on quality–latency trade-offs. For composite emotions, run targeted listener studies with categorical and dimensional (VAD) ratings to verify that mixtures yield recognizable blends rather than cue superposition; test cross-lingual composite control and references with multiple latent emotions.

### Questions
* Data requirements and generalization

How dependent is the method on curated emotional references, and how well do steering vectors transfer across languages, speakers, and acoustic conditions? Clarifying sample needs and robustness to domain shifts would strengthen claims of being training-free and broadly applicable.

* Evaluation methodology and fairness

To what extent do current probes and baselines provide unbiased evidence of effectiveness? A clearer, diversified evaluation (multiple SERs, human studies with rigorous protocols, controlled baseline comparisons) would reduce concerns about circularity and ensure fair positioning against prior work.

* Interpretability and causal validity

The core claim is that a sparse subset of internal tokens encodes emotion and can be causally steered. Can you provide stronger causal tests and time-localization analyses to verify that these tokens specifically modulate emotion (not content or speaker), and relate them to prosodic and VAD dimensions?

* Control stability and safeguards

How do you prevent artifacts and preserve intelligibility/speaker identity at higher steering strengths
alpha (and erasure
beta)? Principles for calibration, adaptive schedules, and clearer safe operating ranges—alongside broader prosodic analyses—would make the control more reliable.
 
* Practicality, efficiency, and compositionality

What are the runtime/memory costs and scalability to streaming/real-time scenarios, and do composite emotions produce predictable, perceptually coherent blends? Demonstrating efficiency trade-offs and validating compositional control would support real-world adoption.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes EmoSteer-TTS, a training-free method to achieve fine-grained, continuous, and interpretable emotion control in pretrained flow-matching TTS models (e.g., F5-TTS, E2-TTS, CosyVoice2). The key idea is activation steering: (1) compute activation differences between neutral and target-emotion references; (2) select top-k emotion-relevant tokens to form a steering vector (and weights); and (3) at inference, add the steering vector with strength alpha to chosen layers/steps to modulate synthesis.

### Strengths
- Works across multiple flow-matching backbones; no fine-tuning required.
- Empirical analyses give actionable guidance: k≈200 works well; multi-layer (spaced) steering outperforms shallow-only; steering across all flow steps is strongest.
- Maintains performance on EMNS/SeedTTS despite steering vectors built from other corpora.
- Low WER and high speaker similarity versus strong flow-matching baselines.

### Weaknesses
- The paper prefers a large alpha but lacks a clear tradeoff curve (alpha vs. WER/N-MOS/E-SIM) and recommended operating range.
- Emotion scores use emotion2vec/SenseVoice; although both are reported, objective metrics can bias toward specific embeddings.

### Questions
- How sensitive are results to the steering corpus composition? Any ablation showing performance when removing one corpus from the construction set?
- If emotion2vec and SenseVoice disagree, which correlates better with MOS? Any human-study correlation numbers to justify metric choices?
- Any comparison with AR-based approach regarding the emotional control? A rough comparison would be great.

### Soundness
3

### Presentation
3

### Contribution
2
