# DMDSpeech: Distilled Diffusion Model Surpassing The Teacher in Zero-shot Speech Synthesis via Direct Metric Optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 8

## Abstract
Diffusion models have demonstrated significant potential in speech synthesis tasks, including text-to-speech (TTS) and voice cloning. However, their iterative denoising processes are inefficient and hinder the application of end-to-end optimization with perceptual metrics. In this paper, we propose a novel method of distilling TTS diffusion models with direct end-to-end evaluation metric optimization, achieving state-of-the-art performance.  By incorporating Connectionist Temporal Classification (CTC) loss and Speaker Verification (SV) loss, our approach optimizes perceptual evaluation metrics, leading to notable improvements in word error rate and speaker similarity. Our experiments show that DMDSpeech consistently surpasses prior state-of-the-art models in both naturalness and speaker similarity while being significantly faster. Moreover, our synthetic speech has a higher level of voice similarity to the prompt than the ground truth in both human evaluation and objective speaker similarity metric. This work highlights the potential of direct metric optimization in speech synthesis, allowing models to better align with human auditory preferences. The audio samples are available at https://dmdspeech.github.io/demo/.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a distilled TTS diffusion model using CTC and SV loss. While the paper is mostly well-written, it lacks novelty and includes some unfair comparisons. Firstly, distilled diffusion models are already well established in computer vision, with a substantial body of research. Even in audio and speech processing, the authors are not among the first to apply this approach, as seen in Nvidia’s blog [1] and other studies. Using SV loss for TTS is also not novel, as it has been previously applied in [3]. Thus, the primary novelty appears to lie in combining these techniques.
The claim of “DIRECT METRIC OPTIMIZATION” is misleading. In speech synthesis, there is no truly direct metric for evaluating generated speech quality. At best, we rely on MOS, while WER and speaker similarity measures serve only as indirect proxies. Generated speech may exhibit machine-like articulation yet achieve low WER. Additionally, speaker similarity measures can be highly domain-dependent and often perform poorly on out-of-domain speakers.
The comparisons made are not entirely fair, given that training with CTC and SV loss naturally results in improved WER and speaker similarity. For RTF evaluation, it seems the authors mainly compare their model with autoregressive models such as XTTS and VoiceCraft, which are not known for their speed. Moreover, the RTF values for DiTTO-TTS and CLaM-TTS are drawn from their respective papers, which is inaccurate since RTF must be measured on the same hardware to be comparable.
NaturalSpeech 3 is not open-source, making it difficult to assess whether its RTF in this paper reflects the original results. A more appropriate comparison would be with StyleTTS2, an open-source, state-of-the-art non-autoregressive model.


[1]https://developer.nvidia.com/blog/speeding-up-text-to-speech-diffusion-models-by-distillation/
[2]Bai, Yatong, et al. "Consistencytta: Accelerating diffusion-based text-to-audio generation with consistency distillation." arXiv preprint arXiv:2309.10740 (2024).
[3]E. Casanova, J. Weber, C. D. Shulby, A. C. Junior, E. Golge, and ¨ M. A. Ponti, “Yourtts: Towards zero-shot multi-speaker tts and zero-shot voice conversion for everyone,” in International Conference on Machine Learning. P

### Strengths
The paper is well written. The experiments are comprehensive although some are not fair comparison.

### Weaknesses
The novelty is limited and the claim of directly optimization is simply not true. Some comparisons in experiments section is not fair.

### Questions
Have the authors try other objective than CTC?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces DMDSpeech, a distilled diffusion model designed for efficient, high-quality zero-shot speech synthesis. It optimizes perceptual metrics by incorporating Connectionist Temporal Classification (CTC) and Speaker Verification (SV) losses, targeting improvements in word error rate (WER) and speaker similarity. The model outperforms previous state-of-the-art approaches in naturalness and speaker similarity, while achieving faster synthesis. This approach highlights the benefits of direct metric optimization in TTS and demonstrates the effectiveness of DMDSpeech in aligning generated speech with human auditory preferences.

### Strengths
**Integration of CTC and SV losses** The paper optimize diffusion TTS model by using end-to-end metric optimization, applying CTC and SV losses to improve word error rate (WER) and speaker similarity.

**Model results** DMDSpeech achieves better performance compared to state-of-the-art baselines with significantly reduced inference time.

### Weaknesses
**Limited Novelty of the Proposed Approach** DMDSpeech utilizes the existing DMD 2 method to distill the teacher model for faster sampling and direct metric optimization, except that DMD 2 simulates a four-step inference process during training, while DMDSpeech simulates a single step. Overall, this paper feels more like an application of DMD 2 in text-to-speech rather than an original methodological advancement.

**Need for Additional Experimental Validation** In DMDSpeech, the student generator is trained to match the teacher model's distribution via distribution matching distillation, but the results in Table 4 indicate that DMD 2 alone decreases MOS-N, SMOS, SMOS-S, and SIM scores compared to the teacher model, with MOS-Q remaining similar. This suggests DMD 2 may offer limited improvement. Why not simply fine-tune the pretrained teacher model with multi-modal adversarial learning and direct metric optimization? Could you conduct additional experiments comparing the teacher model with L_CTC and L_SV​ using GAN against DMDSpeech? Also, please compare the teacher model with L_CTC and L_SV without GAN against DMDSpeech. These would help clarify the impacts of DMD, DMD 2, and GAN in DMDSpeech.

### Questions
1. In line 412, could you elaborate on the statement, "Table 5 shows that our model achieved the highest speaker similarity score (SIM) to the prompt, even surpassing the ground truth"? It is unclear how Table 5 demonstrates that DMDSpeech exceeds the ground truth.

2. Does your training dataset exclude the official samples from the baseline models which are used for evaluation?

3. Could you clarify why you chose to fine-tune the CTC-based ASR model to derive the latent SV model? From your explanation in Appendix C.4, you use a distillation approach to align the CTC-based ASR model's embeddings with the concatenated embeddings from the WeSpeaker's ResNet-based SV model and EPACA-TDNN with a fine-tuned WavLM Large model. Wouldn't it be more efficient to utilize the WeSpeaker's ResNet-based SV model and EPACA-TDNN with a fine-tuned WavLM Large model to directly extract the concatenated embeddings of the prompt and the concatenated embeddings of the generated speech, then calculate the SV loss in Equation 14 using these two embeddings? Can you provide experiment to evaluate that the latent SV model fine-tuned from CTC-based ASR model performs better in speaker similarity?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
In this work, the authors introduce DMDSpeech, a distilled diffusion-based TTS model. In this framework, a student model distills the distribution learned by a diffusion-based teacher model, enabling the generation of high-quality speech in just four steps. By integrating the distillation loss with speaker verification (SV) and connectionist temporal classification (CTC) losses, DMDSpeech achieves excellent speaker similarity and a low word error rate while maintaining high speech quality. The experimental results demonstrate that DMDSpeech outperforms several state-of-the-art baselines as well as the teacher diffusion model.

### Strengths
- This paper is technically sound and well-presented.
- It provides ample implementation details, allowing readers to reproduce the results effectively.
- The authors conduct sufficient experiments to showcase the advantages of diffusion matching distillation, such as faster generation times, and highlight the effectiveness of combining speaker verification (SV) and connectionist temporal classification (CTC) perceptual losses.
- Regarding the mode shrinkage phenomenon resulting from the distillation process, the authors offer a detailed analysis, explaining that this effect may not be undesirable in the context of unconditional TTS.

### Weaknesses
The two major themes of this paper—(1) distillation of diffusion-based TTS models and (2) joint optimization of TTS and perceptual losses—have both been extensively explored in the field of speech generation. While the paper is well-presented and technically sound, the proposed method is relatively straightforward and lacks novelty.

### Questions
The paper is clear enough.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents DMDSpeech, a distilled diffusion model for zero-shot speech synthesis that achieves state-of-the-art performance while significantly reducing inference time. By employing distribution matching distillation, the model generates high-quality speech in just four steps and facilitates direct metric optimization through the use of speaker verification (SV) and connectionist temporal classification (CTC) losses. The authors demonstrate that optimizing these metrics leads to improvements in speaker similarity and word error rate, enhancing the overall intelligibility and quality of synthesized speech. The research highlights the potential of direct metric optimization in bridging the gap between generative modeling and human auditory preferences.

### Strengths
**Originality**
- The authors borrow the concept of distribution matching distillation (DMD) from image synthesis. Yet, the use of DMD in speech synthesis using SV and CTC losses appears to be inspiring and original.
- This should be the first successful application of DMD in speech domain that illustrates promising results.

**Quality**
- As far as I checked, the mathematical derivations are technically sound, and I believe some equations are taken from DMD 2. 
- The training framework designed for zero-shot TTS is also sensible. Expectedly, the CTC loss improves the text alignment and WER, while the SV loss improves the speaker similarity.

**Clarity**
- Section 3 provides comprehensive details about the model architecture, training procedures, and loss functions used, which can aid readers in reproducing the proposed model.
- The results are presented both subjectively and objectively with comparisons to baseline models, which aids in understanding the effectiveness of the proposed model.
- The use of pitch comparison in figure 2 to confirm the improvement of student model is persuasive. This suggests a decent way to analyze the effects of DMD and direct metric optimization.

**Significance**
- DMDSpeech significantly reduces inference time while achieving competitive performance. This could yield a substantial impact to the industry and relative researches.

### Weaknesses
There are some minor issues in this paper. 
- In figure 1, the "subtitle" of the four blocks should be consistently placed (e.g. upper leftmost), otherwise the readers may be difficult to spot them. (I just neglected "Inference", which is inconsistently located at the lower part)
- The choice of four steps in DMDSpeech is not discussed well. The readers may wonder why not 3 or 5 or 20 steps, which both can exhibit significant speedups when compared to 128 steps.

### Questions
Would the quality be better if we use more steps for the student model?

### Soundness
4

### Presentation
3

### Contribution
3
