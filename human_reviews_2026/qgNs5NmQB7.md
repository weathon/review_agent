# TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
We introduce TangoFlux, an efficient Text-to-Audio (TTA) generative model with 515M parameters, capable of generating up to 30 seconds of 44.1kHz audio in 3.7 seconds on a A40 GPU. A key challenge in aligning TTA models lies in creating preference pairs, as TTA lacks structured mechanisms like verifiable rewards or gold-standard answers available for Large Language Models (LLMs). To address this, we propose CLAP-Ranked Preference Optimization (CRPO), a novel framework that iteratively generates and optimizes preference data to enhance TTA alignment. We show that the audio preference dataset generated using CRPO outperforms the static alternatives. With this framework, TangoFlux achieves state-of-the-art performance across both objective and subjective benchmarks. https://tangoflux.github.io/ holds the model-generated audio samples for comparison.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work introduces TangoFlux, a TTA model trained on open data and aligned via CLAP-Ranked Preference Optimization (CRPO). CRPO performs an online, self-improving preference-optimization loop that ranks multiple candidates per prompt with CLAP and optimizes rectified-flow models accordingly.

### Strengths
- Applying online data generation outperforms baselines that use static preference datasets.
- The system can generate up to 30 seconds of audio at 44.1 kHz, an uncommon setting, as most text-to-audio models produce ~10 seconds at 16 kHz.

### Weaknesses
Lacking metrics of FD using PANNs. This metric is the most common metric used by TTA models for evaluating fidelity. (The paper only reports FD using OpenL3)

The evaluation section is missing comparisons to several publicly available state-of-the-art Text-to-Audio models. Specifically, an updated comparison should ideally include results for:
- Make-An-Audio 2 [A]
- EzAudio [B]
- ConsistencyTTA [C]
- AudioLCM [D]

Particularly, ConsistencyTTA and AudioLCM also target on fast inference with consistency models.

[A] Huang, Jiawei, et al. "Make-an-audio 2: Temporal-enhanced text-to-audio generation." arXiv preprint arXiv:2305.18474 (2023).

[B] Hai, Jiarui, et al. "Ezaudio: Enhancing text-to-audio generation with efficient diffusion transformer." arXiv preprint arXiv:2409.10819 (2024).

[C] Bai, Y., Dang, T., Tran, D., Koishida, K., Sojoudi, S. (2024) ConsistencyTTA: Accelerating Diffusion-Based Text-to-Audio Generation with Consistency Distillation. Interspeech 2024

[D] Liu, Huadai, et al. "AudioLCM: Text-to-Audio Generation with Latent Consistency Models." CoRR (2024).

For variable duration, I think it is worth mentioning that the model always operates on a fixed-length latent space, but the duration conditioning explicitly controls how much of that latent space has actual content other than silence. Make sure readers know that this work is not an autoregressive model that can generate an arbitrary length of latent sequence.

Lacking references:

The Related Works section could be strengthened by citing influential but non-public models, such as AudioTurbo [E] and IMPACT [F], to provide a comprehensive overview of the field, similar to how ETTA and Audiobox are referenced. While direct performance comparisons to proprietary models are understandably impossible, acknowledging their existence in the literature is a key academic practice (This will not affect the overall ratings for this work, but it is nice to have).

[E] Zhao, Junqi, et al. "AudioTurbo: Fast Text-to-Audio Generation with Rectified Diffusion." arXiv preprint arXiv:2505.22106 (2025).
[F] Huang, Kuan-Po, et al. "IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling." ICML 2025

### Questions
The batch size used for measuring inference time is not explicitly stated.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a state-of-the-art in-the-wild audio generation model called TangoFlux. The innovations are two-fold:
- A flow-matching model based on FluxTransformer and MMDiT that can generate high-quality audio with variable duration.
- A fine-tuning method called CRPO that alternates between generating preference datasets and using DPO to learn from that dataset.

### Strengths
Overall, I found this paper nice and interesting to read, and I learned a lot from it.

I really like the ablation study between CRPO loss and DPO loss (Figure 3) as well as other settings (Table 3). Super intuitive and informative.

### Weaknesses
The paper is already in good overall shape. That said, there are some remaining weaknesses:
- Human evaluation is performed with 50 human-created and GPT-4o-generated prompts (Section 3.4). This prompt bank seems somewhat small, especially because the 0.2486 OVL z-score of TangoFlux is moderate compared to existing work. Would it be possible to increase the scale of subjective evaluation? If not, some confidence interval or hypothesis testing analyses would greatly clarify the quality advantage of TangoFlux over existing work.

I would also appreciate it if the authors could respond to my questions below.

### Questions
- Which encoder did the KAD metric use? If it is the same encoder as the FD metric, did you find KAD better/more correlated with subjective quality than FD, or worse/less?
- CRPO significantly improves subjective REL. Is there an analysis on what aspects of prompt following have been improved (i.e., temporal relationship of audio events, duration control, etc.)? Do you think conditioning effectiveness can also be improved via CFG?
- TangoFlux allows for duration control. How accurate is this control? Does duration control affect generation quality? Would it be possible to present a scatter plot between the requested duration and the actual generated length?

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
2

### Summary
This study introduces an iterative alignment framework named CRPO to address the challenges Text-to-Audio (TTA) models face in adhering to complex prompts. The method employs a self-improvement loop, leveraging the CLAP model as a proxy reward model to automatically generate and curate preference data. Furthermore, the authors enhance the traditional Direct Preference Optimization (DPO) loss function by incorporating a regularization term to stabilize the training process. Ultimately, they train and open-source a highly efficient and high-performance TTA model called TANGOFLUX based on this method, which demonstrates state-of-the-art performance across multiple evaluations.

### Strengths
The most significant strength of this work is its delivery of a practical and scalable alignment solution for the TTA domain, which lacks off-the-shelf tools. It cleverly repurposes the CLAP model as an effective reward model, addressing the core bottleneck of preference data creation. Technically, the introduction of the LCRPO loss function demonstrates a deep understanding of the potential pitfalls in preference optimization and presents a robust solution. Finally, releasing an open-source, state-of-the-art model trained on public data not only advances the field but also provides an invaluable resource and benchmark for future research.

### Weaknesses
Over-reliance on the Proxy Reward Model: The alignment process is inherently constrained by the capabilities of the CLAP model, as its potential biases and limitations as a proxy reward model could be amplified during iterative training.

Risks of Self-Correction and Limited Generalization Testing: The self-improvement cycle risks reinforcing its own flaws, and the model's generalization is evaluated on a relatively small out-of-distribution dataset, which may not be fully conclusive.

Limited Novelty: The proposed method is not fundamentally novel, as it primarily combines and applies existing, mature technologies to a new domain rather than introducing a new core algorithm.

Lack of Polish in Presentation: The manuscript contains minor typographical errors, such as a quotation mark issue on line 263, which may suggest a need for more thorough proofreading.

The referee acts as the athlete; CLAP RL; CLAP testing.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes text-to-audio (TTA) alignment via DPO. The core idea is to curate a synthetic preference dataset using CLAP: after training an audio generator, the system samples N audios per prompt, selects a winner/loser by CLAP similarity, and forms triplets for DPO. The authors iterate this process and additionally propose keeping a flow-matching loss on winners alongside DPO (denoted 𝐿_CRPO). A flux-like architecture is adopted as the backbone.

### Strengths
- Clear and readable paper.
- The method is intuitive. the evaluation and metrics follows standard evaluation.
- the authors promised to release code and model weights is valuable for the community.

### Weaknesses
**Novelty:** the authors main contribution which is CLAP-driven automatic preference curation for DPO is closely related to Tango-2 and CLIP-DPO as mention in L155-L160, with the main differences being that Tango-2 and CLIP-DPO operates on a static dataset. Running the same loop for multiple iterations seems to me an incremental extension rather than a novel contribution. 

**Evaluation:** the base model is fine-tuned on AudioCaps (L833), which I believe would bias the results on AudioCaps and undermines comparisons to open-source baselines (Stable Audio Open, AudioLDM2). An out-of-distribution evaluation or reporting pre-finetuning numbers for all models would help understanding the effectiveness of the proposed method.

### Questions
- in Figure 2, since CLAP is used to construct preferences, reporting FAD or IS would help understand more the gains.

### Soundness
2

### Presentation
2

### Contribution
2
