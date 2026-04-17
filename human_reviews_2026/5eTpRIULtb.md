# Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Existing dominant methods for audio generation include Generative Adversarial Networks (GANs) and diffusion-based methods like Flow Matching. GANs suffer from slow convergence during training, while diffusion methods require multi-step inference that introduces considerable computational overhead. In this work, we introduce Flow2GAN, a two-stage framework that combines Flow Matching training for learning generative capabilities with GAN fine-tuning for efficient few-step inference. Specifically, given audio's unique properties, we first improve Flow Matching for audio modeling through: 1) reformulating the objective as endpoint estimation, avoiding velocity estimation difficulties when involving empty regions; 2) applying spectral energy-based loss scaling to emphasize perceptually salient quieter regions. Building on these Flow Matching adaptations, we demonstrate that a further stage of lightweight GAN fine-tuning enables us to obtain few-step (e.g., 1/2/4 steps) generators that produce high-quality audio. In addition, we develop a multi-branch network architecture that processes Fourier coefficients at different time-frequency resolutions, which improves the modeling capabilities compared to prior single-resolution designs. Experimental results indicate that our Flow2GAN delivers high-fidelity audio generation from Mel-spectrograms or discrete audio tokens, achieving highly favorable quality-efficiency trade-offs compared to existing state-of-the-art GAN-based and Flow Matching-based methods. Online demo samples are available at \url{https://flow2gan.github.io}, and the source code is released at \url{https://github.com/k2-fsa/Flow2GAN}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents Flow2GAN, a framework that combines flow matching and GAN for efficient one- and two-step neural vocoder generation. It employs a robust flow matching training strategy based on end-point estimation and spectral energy–adaptive loss scaling to achieve high-fidelity waveform synthesis.

### Strengths
[CFM Training Optimization for Robust Waveform Generation]

This paper replaces vector field estimation with end-point estimation to achieve more robust waveform generation. In addition, the loss is scaled by spectral energy. Table 3 demonstrates the effectiveness of the proposed optimization method for high-fidelity waveform synthesis.

However, the analysis in Figure 2 is limited to the spectral domain. While end-point estimation combined with spectral energy–based scaling can improve training efficiency in the early stages and enhance spectral-domain performance, it would be beneficial to further evaluate the model using waveform-level metrics. I understand that standardized waveform-level metrics are lacking, but I recommend including MOS evaluations in Tables 1 and 2 to provide a more comprehensive assessment of perceptual quality.

### Weaknesses
While I like the concept of this paper, my primary concern lies in its lack of novelty.

[End-Point Estimation]

Previous works such as PeriodWave and RFWave have already demonstrated the effectiveness of conditional flow matching for high-fidelity waveform generation. The proposed end-point estimation seems to be a relatively simple modification. Moreover, several recent models, including sCM, have adopted similar end-point estimation schemes for few-step generation.

[Spectral Energy Scaling]

Many works already adopt an energy-based scaling. PriorGrad uses a data-dependent prior based on energy, and they scale the loss with this energy. RFWave also adjusts the loss according to the target's standard deviation. 

Please add further discussion with previous scaling methods. Also, I have a concern about that it only reflect the spectral domain information by removing other information. Did you encounter any issues related to this limitation?

[GAN Fine-tuning]

Honestly, I can not find any difference between your approach and PeriodWave-Turbo which has already shown the efficiency of GAN post-training for CFM-based waveform generation. The paper seems to overclaim novelty in Section 4.2 regarding this aspect.

### Questions
[Q1. Multi-Resolution Network]

Are the inverted outputs from different iSTFT branches simply added together or averaged?

[Q2. Further Steps with GAN Fine-Tuning]

Have you experimented with training four-step or more generators after GAN fine-tuning? It would be interesting to see how performance scales with additional sampling steps.

[Q3. EnCodec Audio Token Experiments]

Could you include PeriodWave-Turbo in Table 2? PeriodWave-Turbo released checkpoints for the same EnCodec-based experiments on their GitHub, so a direct comparison would strengthen your results.

[Q4. TTS Experiments]

Please consider adding TTS results comparing various neural vocoders. Since some vocoders might overfit to ground-truth Mel-spectrograms, conducting two-stage TTS experiments (The generated Mel from TTS Models to waveform) would enhance the quality of the paper.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Flow2GAN introduces a two-stage framework combining Flow Matching (FM) pre-training and GAN fine-tuning to balance audio generation quality and inference speed. Key innovations include endpoint-prediction-based FM (to handle silent regions), spectral energy-adaptive loss scaling (for perceptual alignment), and a multi-resolution ConvNeXt network. Experiments show competitive performance against SOTA models, though critical gaps exist in comparisons to key competitors and hybrid FM+GAN baselines.

### Strengths
Hybrid Paradigm: Effectively merges FM’s stable training with GAN’s detail refinement, resolving FM’s slow inference and GAN’s mode collapse issues.
Audio-Specific FM Improvements: Endpoint prediction and energy-adaptive loss significantly boost FM’s performance, validating their utility for audio synthesis.
Strong Experimental Results: Outperforms Vocos, RFWave, and WaveFM on most metrics (Table1/2) for Mel-spectrogram and Encodec token conditioning.
Multi-Resolution Network: Enhances frequency modeling compared to single-resolution designs (Table5), improving perceptual audio quality.

### Weaknesses
Training Complexity: Two-stage training (FM pre-training + GAN fine-tuning) increases the barrier to reproduction and deployment—users need to manage separate pipelines and hyperparameters for each stage.

Incomplete and Unconvincing Comparison Landscape:
a. Lack of coverage of GAN-enhanced Flow Matching models: The paper claims novelty in its hybrid FM+GAN framework, but many existing models leverage GANs to accelerate Flow Matching. A detailed comparison to these models—including their design choices, performance, and efficiency—is missing, which obscures Flow2GAN’s unique contributions and relative standing.
b. Performance does not exceed BigVGAN: Even accounting for dataset size differences (BigVGAN uses a larger dataset), the paper’s reported metrics (e.g., PESQ, audio quality scores) do not demonstrate that Flow2GAN outperforms BigVGAN. The claim of being "state-of-the-art" is thus unsubstantiated against this key competitor.
c. Significant speed gap with Vocos: The paper emphasizes inference efficiency but fails to address that its speed is much slower than Vocos (a leading efficient audio generation model). This gap undermines the practical utility of Flow2GAN for real-world applications where low latency is critical.

### Questions
NA

### Soundness
2

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
This paper introduces Flow2GAN, a two-stage hybrid framework for high-fidelity neural vocoding that aims to combine the training stability of Flow Matching with the detail-refinement capabilities of GANs. The authors first propose improvements to the Flow Matching stage tailored for audio, including reformulating the objective to direct endpoint prediction and introducing a spectral energy-adaptive loss scaling to focus on perceptually important regions. Subsequently, a pre-trained model is used to initialize one- and two-step generators, which are then fine-tuned using adversarial training. The architecture is a multi-resolution network that processes spectral coefficients at different time-frequency scales. Experiments on both mel-spectrogram and audio token conditioning show that Flow2GAN achieves state-of-the-art results in terms of both audio quality and inference efficiency.

### Strengths
The paper's core concept of a two-stage training paradigm is well-motivated and presents a clever solution to a known trade-off in generative modeling. The approach logically leverages Flow Matching for robust, global structure learning and then uses a fast GAN fine-tuning stage for refining high-frequency details, which is an effective strategy. The proposed modifications to the Flow Matching objective appear sound; the shift to endpoint prediction is an intuitive way to handle silent regions in audio, and the ablation studies convincingly demonstrate its benefits. The experimental results are another clear strength. The model achieves impressive scores across multiple metrics and tasks, outperforming several strong baselines while offering significantly faster inference than multi-step diffusion models.

### Weaknesses
Despite the strong results, the paper has several weaknesses in its positioning and methodological clarity that should be addressed. First, the proposed "spectral energy-adaptive loss scaling" is conceptually very similar to the "energy balanced loss" used in prior work like RFWave, yet the paper fails to discuss, or compare against it. This omission makes it difficult to assess the novelty of this specific contribution. Second, the reformulation of the prediction target from velocity to endpoint is a significant change to the underlying probability flow ODE. The paper presents a modified sampling equation but does not provide a clear derivation or justification for it, leaving a gap in the methodological explanation.
Furthermore, a key concern with any GAN-based method is the risk of mode collapse. The paper claims that its two-stage approach mitigates this risk, but this assertion is made without any supporting evidence, either quantitative (e.g., diversity metrics) or qualitative. Finally, while the paper emphasizes its fast one/two-step inference, it critically lacks comparisons to other state-of-the-art fast sampling methods for flow and diffusion models, such as consistency models or recent shortcut models, which are the most relevant competitors for few-step generative performance.

### Questions
To help clarify the contributions and rigor of the paper, I would appreciate the authors' response to the following:
1. Could you please elaborate on the relationship between your proposed spectral energy-adaptive loss scaling and the energy balanced loss from RFWave? How does your approach differ, and what are its specific advantages that lead to the observed performance gains?
2. When you reformulate the training objective to endpoint prediction (Equation 4), the underlying ODE changes. Could you provide a more thorough derivation or explanation for the new sampling process?
3. You claim that the Flow Matching pre-training mitigates the risk of GAN mode collapse. This is a significant claim. Could you provide any empirical evidence to support it, for instance, by analyzing the output diversity of Flow2GAN compared to a pure GAN baseline trained for a similar duration?
4. The main appeal of the model is high-quality generation in one or two steps. Why were there no comparisons against other prominent few-step or single-step generative model sampling strategies, such as consistency models or recent "shortcut models" (e.g., as proposed by Frans et al., 2024), which are designed to solve the exact same problem?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Flow2GAN is a two-stage framework for high-fidelity audio generation that integrates Flow Matching and GAN. It first uses improved Flow Matching to learn robust generative capabilities, which reformulated as endpoint estimation to avoid velocity prediction issues in empty audio regions and enhanced with spectral energy-adaptive loss scaling to emphasize perceptually important quiet areas. Then, lightweight GAN fine-tuning refines details, enabling efficient one- or two-step inference. Equipped with a multi-resolution network processing Fourier coefficients at different time-frequency resolutions, it outperforms state-of-the-art GAN and Flow Matching-based methods in both quality and efficiency under Mel-spectrogram and Encodec audio token conditioning.

### Strengths
1. The two-stage design effectively combines the stable training of Flow Matching and the efficient fine-grained generation of GAN, addressing the slow convergence/mode collapse of GANs and high computational cost of diffusion methods.
2. For audio’s unique properties, the authors propose endpoint estimation and spectral energy-adaptive loss scaling to improve Flow Matching, significantly enhancing generation quality in silent regions and perceptual consistency.
3. The multi-resolution network structure outperforms single-resolution designs in modeling audio complexity, providing a powerful backbone for generative learning.

### Weaknesses
1. Compared to BigVGAN-v2 trained on a larger dataset, it still has a slight gap in some metrics, suggesting limitations in generalization to larger-scale data.
2. The one-step model’s performance at low bandwidth (1.5 kbps) is inferior to its two-step version and some competitors, leaving room for improvement in low-bandwidth audio generation.

### Questions
1. What is the majoy difference between the proposed method and PeriodWave-Turbo? Is it just the improved Flow Matching model?
2. Can this improved Flow Matching strategy be applied to text to speech/audio tasks?

### Soundness
3

### Presentation
3

### Contribution
3
