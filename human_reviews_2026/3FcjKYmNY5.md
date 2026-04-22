# SALSA-V: Shortcut-Augmented Long-form Synchronized Audio from Videos

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
We propose SALSA-V, a multimodal video-to-audio generation model capable of synthesizing highly synchronized, high-fidelity long-form audio from silent video content. Our approach introduces a masked diffusion objective, enabling audio-conditioned generation and the seamless synthesis of audio sequences of unconstrained length. Additionally, by integrating a shortcut loss into our training process, we achieve rapid generation of high-quality audio samples in as few as eight sampling steps, paving the way for near-real-time applications without requiring dedicated fine-tuning or retraining. We demonstrate that SALSA-V significantly outperforms existing state-of-the-art methods in both audiovisual alignment and synchronization with video content in quantiative evaluation and a human listening study. Furthermore, our use of random masking during training enables our model to match spectral characteristics of reference audio samples, broadening its applicability to professional audio synthesis tasks such as Foley generation and sound design.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a new video-to-audio model called SALSA-V that efficiently generates long-form audio given a target video. To perform long-form audio generation, the proposed method iteratively outpaints the previously generated audio, and for that purpose, the model is trained with masked ground-truth audio as conditional input. For efficient generation, the training objective includes a shortcut loss, which enables few-step generation during inference. The experimental results show that the proposed model outperforms existing models, especially when the duration of the target video is long or the number of sampling steps is small.

### Strengths
- The proposed method is simple and should be easy to implement.
- The advantage of the proposed method over MMAudio is significant when the duration of the target video is long or the number of sampling steps is small.

### Weaknesses
- The novelty of the methodology is marginal.
  - The main modifications from MMAudio are training with masked audio and the use of shortcut loss, and both techniques appear to be adopted in a straightforward manner.
  - It would be beneficial to clearly describe the particular challenges in applying these techniques to video-to-audio models and how the proposed method addresses them. Alternatively, the authors can mention any empirical insights that could be helpful for future studies in the field of video-to-audio generation.
- The comparison in the experiments needs additional baselines.
  - For efficient generation, Frieren (or applying reflow to the proposed model with a flow matching formulation), as mentioned in Section 2.3, would be a good baseline.
  - For long-form generation, V-AURA (mentioned in Section 2.1) and LoVA (mentioned in Section 2.4) could be used as baselines.
- The benefit of audio-conditional generation in SALSA-V is not clear in Figure 4.
  - I understand that some audio patterns in the conditional audio faithfully appear in the generated audio, but it seems that the ground-truth audio does not actually contain such patterns within the generated timeline. Thus, it is not clear if the conditional audio contributes to accurate audio generation.
- I could not access the demo page due to a timeout error.

### Questions
- Is there any reason why the authors did not try SALSA-V with 1B parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents SALSA-V, a shortcut-augmented latent flow matching model for video-to-audio generation that achieves high-fidelity and temporally synchronized audio in just a few sampling steps. By combining masked training for audio conditioning and outpainting with contrastively trained synchronization features, SALSA-V enables both efficient short-form and stable long-form audio generation without distillation.

### Strengths
**1. Enables Long-Form Audio Generation**

The model supports audio-conditioned outpainting, allowing stable and coherent generation of extended audio sequences (30 s+) from video inputs.

**2. Improved Audio-Visual Synchronization**

A contrastively trained synchronization encoder provides precise temporal alignment between visual motion and audio events, achieving SOTA synchronization performance.

**3. Efficient Few-Step Sampling**

The shortcut-augmented flow matching formulation reduces sampling steps to ≤ 8 without quality degradation.

### Weaknesses
**1. Low Readability and Clarity**

The paper’s presentation is occasionally hard to follow.

**2. Inconsistency between Quantitative Results (Fig. 5 vs. Table 1)**

There appears to be a mismatch between the 10s performance trend in Figure 5 and the metrics in Table 1.

**3. Insufficient Analysis of Inference Efficiency**

While the paper emphasizes few-step generation, there is no thorough empirical analysis of inference speed. More concrete runtime results would strengthen the efficiency claims.

### Questions
**1. Clarification on Feature Roles and Figure References**

In the Method section, the paper sequentially introduces the VAE encoder, visual-text representation, and synchronization feature.
However, it is unclear how each of these corresponds to the components shown in Figure 1, and what specific roles they play in the generation pipeline. A clearer mapping between the described features and their positions in Figure 1 would be helpful.
In addition, Figure 5 is never explicitly referenced in the main text—please consider mentioning and explaining it within the corresponding experimental section.

**2. Inconsistency between Figure 5 and Table 1 Results**

In Figure 5, the proposed model shows lower performance than MMAudio for 10-second generation, whereas Table 1 indicates a different trend. Could you clarify whether these results are based on different experimental setups or evaluation protocols, and explain what accounts for the discrepancy?

**3. Quantitative Analysis of Sampling Efficiency**

The paper highlights sampling efficiency as a major advantage of SALSA-V, but does not provide concrete runtime comparisons.
Could the authors include or discuss actual inference speed measurements (e.g., seconds per 10-second clip, GPU type, batch size), and how they compare to other models?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces SALSA-V, a model for video-to-audio (V2A) generation, designed to synthesize high-fidelity, long-form audio that is precisely synchronized with video content. The authors propose three main contributions: (1) a shortcut-augmented training objective to enable high-quality audio generation in very few sampling steps; (2) a masked flow matching approach that allows the model to perform audio-conditioned generation and outpainting, thereby enabling the creation of audio for long-form videos through iterative extension; and (3) a new contrastive audio-visual synchronization model built upon a strong pre-trained vision backbone to yield high-resolution alignment features.

### Strengths
*   **Clear Presentation and Structured Evaluation**: The paper is clearly written, with a well-defined problem motivation and a structured presentation of its methods. The evaluation is methodical, employing a range of established objective metrics alongside a human listening study.

*   **Focus on Practical V2A Challenges**: The work addresses relevant practical limitations in the video-to-audio (V2A) domain, particularly the efficiency of the sampling process and the synthesis of audio for longer-form videos. This focus is pertinent to improving the applicability of such generative models.

*   **Achieves Strong Temporal Synchronization**: A notable strength of the proposed model is its ability to generate tightly synchronized audio. The paper reports state-of-the-art results on the DeSync metric, and this quantitative improvement in temporal alignment is also corroborated by the human evaluation study.

### Weaknesses
The paper, while presenting a well-engineered system, suffers from several weaknesses that question the significance and novelty of its contribution.

1.  **Limited Novelty and Incremental Contribution**: The primary weakness of this work is its reliance on existing techniques across its entire pipeline, from the architectural framework and training methods to the experimental conclusions. This makes the overall contribution feel incremental rather than innovative.
    *   The model's core design, which combines semantic and high-resolution synchronization features for conditioning, directly follows the framework established by **MMAudio[1]**.
    *   The use of a masked training objective for audio conditioning is a known technique, with similar approaches having been explored in works like **AudioX[2]** and **MultiFoley[3]**.
    *   The key insight regarding the choice of batch size for training the synchronization model is also acknowledged to be a direct replication of findings from **Synchformer [4]**.
    While the integration of these parts is functional, the paper does not introduce a new fundamental concept, algorithm, or a significant insight to the field.

2.  **Insufficient Experimental Baseline and Missing Citations**: The paper's experimental comparison is narrow, failing to benchmark against a sufficient number of relevant contemporary models. This makes it difficult to accurately assess its performance in the broader context of the field.
    *   The main quantitative comparison (Table 1) only includes two other models, **FoleyCrafter** and **MMAudio**. Other highly relevant V2A methods, such as the autoregressive model **V-AURA [5]** and other diffusion-based models like **AudioX** and **Frieren[6]**, are not included in the benchmark.
    *   A more comprehensive comparison against a wider array of recent models is necessary to robustly support the claims of state-of-the-art performance.

3.  **Contradictory Results and Lack of Significant Improvement**: Several claims in the paper are not fully supported by its own results, and the overall improvement is not compelling.
    *   **Lower Subjective Audio Quality**: For a generative model, perceptual quality is paramount. The human evaluation in Table 1 shows that SALSA-V's **Audio Quality** score (2.96) is lower than the baseline MMAudio (3.16). This key result undermines the paper's claim of outperforming existing methods. The overall subjective improvement is not significant.
    *   **Potentially Flawed Long-Form Evaluation**: The paper compares its iterative, chunk-based generation with MMAudio's one-shot, full-sequence approach. As the models operate with different context windows and inference strategies, this direct comparison of metrics could be misleading.
    *   **Disconnect Between Main Contribution and SOTA Results**: The "shortcut loss" for few-step sampling is highlighted as a major contribution. However, the main results in Table 1, which establish the model's SOTA synchronization, are based on 32 sampling steps. The paper does not demonstrate that this claimed SOTA performance is retained in the few-step regime, thus disconnecting the novel claim from the primary comparative results.

4.  **Lack of Clarity in Experimental Details and Reproducibility Issues**: The description of the experimental setup lacks the necessary detail for reproducibility.
    *   The test set used for evaluation is vaguely described as a composition of "a holdout set of in-the-wild videos, the VGGSound test set, and UnAV-100". Without precise details on the composition, data splits, and preprocessing of this custom benchmark, the results are not verifiable or reproducible by the community.

[1] Cheng H K, Ishii M, Hayakawa A, et al. MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 28901-28911.

[2] Tian Z, Jin Y, Liu Z, et al. Audiox: Diffusion transformer for anything-to-audio generation[J]. arXiv preprint arXiv:2503.10522, 2025.

[3] Chen Z, Seetharaman P, Russell B, et al. Video-guided foley sound generation with multimodal controls[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 18770-18781.

[4] Iashin V, Xie W, Rahtu E, et al. Synchformer: Efficient synchronization from sparse cues[C]//ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024: 5325-5329.

[5] Viertola I, Iashin V, Rahtu E. Temporally aligned audio for video with autoregression[C]//ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025: 1-5.

[6] Wang Y, Guo W, Huang R, et al. Frieren: Efficient video-to-audio generation network with rectified flow matching[J]. Advances in Neural Information Processing Systems, 2024, 37: 128118-128138.

### Questions
1.  **On Novelty**: The paper's core components (architecture, masked training, batch size insights) appear to be adapted from prior work. Could the authors clarify the primary technical novelty beyond the successful integration of these known techniques?

2.  **On Experimental Baselines**: The experimental comparison is limited. Could the authors justify the exclusion of several recent and relevant baselines like V-AURA, AudioX, and Frieren, and perhaps provide a comparative analysis on key metrics?

3.  **On Few-Step Generation Performance**: The "shortcut loss" is a key claimed contribution, but SOTA results are shown at 32 steps. Could the authors provide key metrics (e.g., DeSync, FAD) for the 8-step generation case against baselines to demonstrate the practical effectiveness of this feature?

4.  **On Reproducibility**: The evaluation benchmark is vaguely described. For reproducibility, could the authors provide a precise composition and list of identifiers for their custom test set?

### Soundness
4

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SALSA-V, a novel video-to-audio generation model that synthesizes high-fidelity, temporally aligned long-form audio from silent videos. The model leverages a masked diffusion objective to support audio-conditioned generation and seamless outpainting, enabling the synthesis of audio sequences of arbitrary length. A key innovation is the integration of a shortcut loss during training, which allows for high-quality audio generation in as few as eight sampling steps without additional fine-tuning. Furthermore, the authors introduce a contrastively-trained synchronization module using a large-scale pretrained vision backbone, which significantly improves temporal alignment between visual events and generated sounds. Extensive evaluations demonstrate that SALSA-V outperforms existing state-of-the-art models in both objective metrics (e.g., DeSync, FAD) and human subjective ratings, particularly in synchronization and long-form generation quality. The model also maintains competitive performance in semantic alignment and audio fidelity, despite having fewer parameters than leading baselines. These contributions collectively address key limitations in current V2A systems, including generation speed, controllability, and scalability to long durations.

### Strengths
* High-Quality Synchronization: The model demonstrates state-of-the-art performance in temporal alignment between audio and video, as evidenced by both objective metrics (DeSync) and human evaluation. This is a critical and challenging aspect of video-to-audio generation.
* Efficient Few-Step Sampling: By incorporating a shortcut loss during training, the model achieves high-quality audio generation in as few as eight sampling steps, without requiring additional fine-tuning or distillation. This makes it suitable for near-real-time applications.
* Long-Form Generation Support: Through a masked diffusion objective, the model supports audio conditioning and seamless outpainting, enabling the generation of synchronized audio for extended video sequences without significant performance degradation.
* Strong Multimodal Conditioning: The model effectively leverages multiple conditioning sources—semantic visual features, high-resolution synchronization features, and text embeddings—using a modified multimodal transformer architecture, leading to improved semantic and temporal alignment.

### Weaknesses
* Evaluation of Contrastive Pre-training Models: The CAVP model from Diff-Foley is also a pre-trained model for audio-visual alignment, yet the authors omitted a comparative analysis with it. Could an experiment be designed to compare the performance of these two models? Additionally, while the paper discusses the impact of different batch sizes on contrastive learning, it lacks an ablation study to substantiate these claims.
* Masked Training Strategy: During inference, only the preceding segment of clean audio is prepended. Why, then, is the masking during training applied randomly rather than being restricted to the preceding portion as well? Is there any prior investigation or ablation experiment regarding this design choice?

### Questions
Please refer to the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3
