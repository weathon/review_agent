# Unified Vision–Language Modeling via Concept Space Alignment

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
We introduce V-SONAR, a vision–language embedding space extended from the
text-only embedding space SONAR (Omnilingual Embeddings Team et al., 2026),
which supports 1500 text languages and 177 speech languages. To construct
V-SONAR, we propose a post-hoc alignment pipeline that maps the representations
of an existing vision encoder into the SONAR space. We thoroughly evaluate
V-SONAR and show that its embeddings achieve competitive performance on
text-to-video retrieval. Equipped with the OMNISONAR text decoder, V-SONAR
further surpasses state-of-the-art vision–language models on video captioning tasks,
including DREAM-1K (BLEU 23.9 vs. 19.6) and PE-VIDEO (BLEU 39.0 vs. 30.0).

Leveraging V-SONAR, we first demonstrate that the Large Concept Model (LCM;
LCM team et al. 2024) operating in SONAR and trained with English text only, can
perform both single- and multi-visual concept understanding in a zero-shot manner.
Finally, we introduce V-LCM, which extends the LCM with vision–language
instruction tuning. V-LCM encodes vision and language inputs into an unified
sequence of latent embeddings via V-SONAR and SONAR, and it is trained with
the same latent diffusion objective for next-embedding prediction as in LCM’s
text-only pre-training. Experiments on a large-scale multilingual and -modal
instruction–tuning data mixture highlight the potential of V-LCM: V-LCM matches
state-of-the-art vision-language models on tasks covering image/video captioning
and question answering, while significantly outperforming them across 61 rich- to
low-resource languages out of all 62 tested languages.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces v-Sonar, a paradigm to map vision representation into Sonar’s embedding space via representation alignment, enabling direct reasoning and decoding in a shared latent space. v-Sonar achieved strong zero-shot text-to-video retrieval and video captioning performance. Building upon this, the paper introduces v-lcm, a multi-modal version of large concept models, which perform latent diffusion over unified visual-text latent space, and achieves comparable performance to SOTA VLMs.

### Strengths
1. The paper is well-written and easy to follow.
2. The evaluation and ablation studies are very comprehensive.
3. v-Sonar and v-lcm present great zero-shot capabilities, which are very interesting.

### Weaknesses
1. The paper is more like a technique report with limited technical contributions. The core concept, aligning vision encoders to text encoders, has been proposed by previous work [1]. The specific techniques to align visual-textual representations (MSE and contrastive loss) are also very standard.
2. Most metrics are only comparable to SOTA models, except for the multilingual ability, which may partly be due to the fact that Sonar is specially designed and trained with specific datasets for this objective. While it is encouraging to explore architectures different from current LLMs even with temporary underperformance, it is preferable that the new model can demonstrate significant advantages in certain special aspects.
3. Some of the implementation details are unclear. See questions.

[1] Jose, Cijo, et al. "Dinov2 meets text: A unified framework for image-and pixel-level vision-language alignment." CVPR 2025.

### Questions
1. The open-sourced PE-VIDEO dataset contains around 120k videos with annotation. However, v-Sonar uses 200k video-caption pairs to refine the alignment in stage 3 and uses 15k pairs for evaluation. Are there any overlaps?

2. How is v-lcm supervised finetuned? Do you update all the parameters?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper extends the SONAR multilingual embedding space to vision via V-SONAR, which aligns a pretrained Perception Encoder to SONAR through a post-hoc MSE-based mapping and a three-stage image/video captioning curriculum. This enables the Large Concept Model (LCM), trained only on text embeddings, to operate directly on visual inputs. The authors further introduce V-LCM, a vision–language instruction-tuned version trained in the shared SONAR/V-SONAR latent space. The approach achieves zero-shot video retrieval and captioning, and V-LCM matches or surpasses state-of-the-art models on multilingual vision–language tasks across most of the languages.

### Strengths
1. The paper discussed a clear and well-motivated idea of extending the SONAR embedding to visual modalities through a lightweight post-hoc alignment.
2. The proposed alignment pipeline is designed well and experimentally validated.

### Weaknesses
1. The paper’s main idea, while clearly implemented, feels somewhat incremental as it builds on existing post-hoc alignment frameworks rather than introducing a fundamentally new alignment principle.

2. The evaluation mainly focuses on retrieval and captioning benchmarks, which measure semantic alignment but not grounding quality. It remains unclear whether V-SONAR preserves sufficient spatial and compositional information for tasks that require reasoning about object relations, layouts, or fine-grained attributes.

3. The reported numerical gains cannot be attributed solely to the proposed alignment, since the results combine the benefits of the stronger SONAR2 text space and the powerful Perception Encoder backbone. A more controlled comparison, such as aligning to both SONAR1 and SONAR2, or directly comparing against the unaligned vision encoder, would better isolate the contribution of the alignment itself.

### Questions
- Can the authors report a controlled comparison, such as aligning to SONAR1 versus SONAR2 or evaluating against the unaligned Perception Encoder, to quantify the gain contributed by the proposed alignment?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces V-SONAR, a framework that aligns visual encoders into the SONAR multilingual concept space, and extends the Large Concept Model (LCM) to vision–language tasks via V-LCM.
Instead of joint multimodal training, the authors propose a post-hoc alignment of frozen visual features to an existing multilingual embedding space, enabling zero-shot and multilingual image/video understanding.
Extensive experiments on retrieval, captioning, and multilingual instruction following show competitive or superior results, especially across 61 of 62 evaluated languages.

### Strengths
- **Elegant unification:** The method merges vision and language in a shared continuous embedding space, avoiding heavy multimodal retraining.

- **Practical efficiency:** Simple MSE-based alignment achieves strong zero-shot and multilingual performance with modest compute.

- **Comprehensive evaluation:** Covers image/video tasks, long-video summarization, and multilingual benchmarks.

- **Insightful analysis:** Provides embedding-geometry diagnostics (AC/trace/logdet) to interpret cross-modal alignment quality.

- **Strong empirical results:** Outperforms major baselines like SigLIP2 and PLM on several metrics.

### Weaknesses
- **Lack of clear conceptual  motivation** –
The paper reads more like a comprehensive technical report. While the system is well engineered, the underlying motivation and novelty are not sharply defined. The central idea—post-hoc alignment of visual features to a frozen multilingual embedding space—feels incremental, relying on existing components (SONAR, LCM).

- **Potential semantic collapse within the aligned subspace.**
Aligning visual features via an MSE loss to a frozen text space risks over-compression: different visual scenes may map to similar embeddings. The paper lacks an investigation into embedding diversity or anisotropy after alignment—important for understanding retrieval and generation limits.

- **No examination of cross-modal consistency at inference.**
The paper shows that visual inputs can be decoded by the LCM text generator, but doesn’t analyze whether the generated language faithfully represents the same semantic region of the embedding space. Cross-modal drift—where decoding shifts meaning—may exist but is untested.

### Questions
- When decoding visual inputs into text via LCM, have you evaluated whether the generated descriptions remain semantically faithful to the corresponding visual embeddings? If not, how do you assess or mitigate possible cross-modal drift?
- Could you include some qualitative or more interpretable examples (e.g., generated captions, retrieval results, or typical errors) to illustrate how the model behaves in practice?
- Could the post-hoc alignment to a fixed text space cause semantic drift or information loss, especially for fine-grained visual details that have no direct linguistic counterpart?

### Soundness
3

### Presentation
3

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
This paper introduces v-SONAR, a vision-language embedding space constructed by post-hoc aligning a state-of-the-art vision encoder with the existing multilingual text embedding space Sonar. The authors propose a three-stage curriculum that leverages large-scale image-caption and video-caption datasets to align Perception Encoder representations into Sonar's semantic concept space, supporting 200 text languages and 37 speech languages in addition to images and videos. Thorough experiments evaluate the alignment and demonstrate competitive or superior zero-shot performance in text-to-video retrieval, video captioning, and especially in multilingual and multi-modal instruction following. The framework is further extended with v-LCM, a latent-diffusion-based unified vision-language model, which is instruction-tuned and evaluated across a large suite of language and vision-language downstream tasks.

### Strengths
1.Instead of training a massive multilingual VLM from scratch, the authors cleverly "inherit" the advantages of two SOTA models: SONAR's powerful multilingual semantic space and PERCEPTION ENCODER's strong visual representation capabilities. This approach is more computationally efficient and highly scalable.

2.By modeling in the unified SONAR/V-SONAR latent space, v-LCM successfully generalizes its visual understanding capabilities to a massive number of low-resource languages, significantly outperforming strong baselines like Qwen-VL and PLM-8B in 61 out of 62 languages.

3.The proposed Large Concept Model (LCM), trained only on text, is capable of directly processing V-SONAR encoded visual embeddings and performing complex tasks like video summarization without having seen any visual data. This strongly demonstrates that V-SONAR has successfully aligned visual concepts to the SONAR concept space at a semantic level.

4.The alignment method employs a three-stage coarse-to-fine curriculum, progressively achieving visual-semantic alignment. This training strategy demonstrates good engineering feasibility.

### Weaknesses
1.Critical Flaw: Inconsistent and Non-Reproducible SONAR Versions. The paper's two main SOTA (state-of-the-art) achievements are based on two different, incompatible SONAR versions. The SOTA video captioning (Table 3) uses SONAR2, an unreleased, higher-performing internal model . Conversely, the flagship multilingual SOTA results (Fig. 3) rely on the public SONAR1 , as the LCM was trained on it. This mismatch creates significant confusion about the paper's true contribution and renders the SOTA video captioning results non-reproducible.

2.The paper introduces v-LCM as a "new paradigm" built upon the Large Concept Model (LCM), which is a latent diffusion model . Unlike standard Transformers, diffusion models require multiple iterative sampling steps for generation, which is typically orders of magnitude slower. The paper provides no discussion on inference latency or computational cost, making it impossible to evaluate the practical utility of v-LCM against standard auto-regressive VLMs.

3.Lack of Analysis on Training Data Bias. The 12M+ image and video pairs used for alignment are likely English-centric. The paper provides no analysis of this potential data bias or how it impacts the model's outstanding multilingual generalization. It is unclear if this capability is learned during alignment or simply a "free lunch" inherited from the pre-trained SONAR space.

4.Multilingual Generalization vs. English SOTA. While the multilingual results are exceptional, the paper's own data (Table 5) shows that v-LCM is not the overall best on competitive English benchmarks. It trails the PLM-8B baseline in tasks like video summarization (VIDEOXUM) and document QA (VisualMRC). This indicates the model's core strength is its broad multilingual generalization rather than peak performance in high-resource English tasks.

5.The paper justifies its use of MSE loss over a standard contrastive loss by claiming "no significant gains" . However, the data in Appendix B (Table 6) shows a clear trade-off: contrastive loss performed better on retrieval (R@1 52.4 vs. 49.0) but slightly worse on captioning (BLEU 38.6 vs. 38.9). The paper lacks a necessary technical analysis of why this trade-off exists and why MSE is the better choice for generative alignment.

### Questions
1. The paper's two main SOTA achievements are based on two different SONAR versions: SONAR2 for video captioning and SONAR1 for multilingual v-LCM . Given that SONAR2 is reported to be substantially stronger , does this imply that the multilingual performance of v-LCM (based on SONAR1) is significantly underestimated? What is the expected performance ceiling for v-LCM if it were aligned and trained with SONAR2?

2. v-LCM is based on a latent diffusion model (LCM) , which requires iterative denoising for generation. This process is typically much slower than the autoregressive transformers used in baseline VLMs. What is the actual inference latency of v-LCM compared to models like Qwen-VL or PLM-8B, and why was this critical discussion of computational cost omitted when assessing its practical utility as a "new paradigm"?

3. v-LCM achieves success in 61 languages , yet its alignment data (12M images + 2M videos ) is likely English-dominated. How does the paper prove that v-LCM learned a true alignment between visual concepts and non-English concepts, rather than simply "inheriting" this capability as a "free lunch" from the pre-trained SONAR space?

4. Appendix B shows that contrastive loss was superior for retrieval (52.4 vs. 49.0), while the chosen MSE loss was only marginally better for captioning (38.9 vs. 38.6). Why does the more common contrastive loss, which improves retrieval alignment, appear to hurt the performance of generative tasks like captioning? Can a deeper technical analysis be provided?

5. Why did v-LCM fail to outperform baselines in Dutch and lag significantly on English tasks like VisualMRC (Table 5)? Does this expose a fundamental limitation of the v-LCM framework in handling specific languages or task types?

### Soundness
3

### Presentation
3

### Contribution
2
