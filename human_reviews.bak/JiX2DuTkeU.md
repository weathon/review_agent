# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

- Decision: Reject
- Scores: 8, 6, 3, 5

## Abstract
This paper introduces F5-TTS, a fully non-autoregressive text-to-speech system based on flow matching with Diffusion Transformer (DiT). Without requiring complex designs such as duration model, text encoder, and phoneme alignment, the text input is simply padded with filler tokens to the same length as input speech, and then the denoising is performed for speech generation, which was originally proved feasible by E2 TTS. However, the original design of E2 TTS makes it hard to follow due to its slow convergence and low robustness. To address these issues, we first model the input with ConvNeXt to refine the text representation, making it easy to align with the speech. We further propose an inference-time Sway Sampling strategy, which significantly improves our model's performance and efficiency. This sampling strategy for flow step can be easily applied to existing flow matching based models without retraining. Our design allows faster training and achieves an inference RTF of 0.15, which is greatly improved compared to state-of-the-art diffusion-based TTS models. Trained on a public 100K hours multilingual dataset, our Fairytaler Fakes Fluent and Faithful speech with Flow matching (F5-TTS) exhibits highly natural and expressive zero-shot ability, seamless code-switching capability, and speed control efficiency. Demo samples can be found at https://F5-TTS.github.io. We will release all code and checkpoints to promote community development.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This study uses flow-matching based diffusion model for TTS. As opposed to the earlier studies which usually require additional modules such as an alignment model (e.g. in Voicebox), the proposed model pads the character sequence up to the length of the speech features and uses a ConvNext-like model to let the text features align with speech during the training process. The paper also proposes to use Sway sampling during model inference. This strategy samples the timestep non-uniformly. According to the experiments on Librispeech and a Chinese set, the proposed model results in comparable WER on the test data with previous models (Voicebox and E2-TTS). The RTF also improves in the proposed model.

After reading the rebuttal, I would like to maintain my score.

### Strengths
Originality, 

- Sufficiently original: The paper builds on the idea of E2-TTS which pads the character sequence up to the number of speech feature frames and with different modeling choices (transformer instead of Unet, and ConveNext based text frontend), the proposed system can handle better speech and text alignment. 

- The paper also proposes Sway sampling strategy to the inference instead of uniform time step sampling. The paper suggests that one advantage of this strategy is that it can prevent speaker information leaks in the cases where a speaker or context change occurs.

- The way the paper uses time step representation in the training setup is different than prior studies.

Quality, 

- Several ablation experiments are performed. Some of those results are included in the Appendix. 


Clarity,

- Writing is clear.

- Experimental setup details are mentioned in detail.  


Significance

- Not requiring additional modules (e.g. a duration model) is an advantage of the proposed method.

### Weaknesses
1. One problem in experimental settings is the lack of direct comparison between Voicebox, E2-TTS and the current study. The paper already mentions that the prompting test sets are not known and hence it is hard to compare, a reproduction of the setups could have solved this problem. In the Appendix, it seems that there are some comparisons with E2. It would be better to include some of those comparisons in the main text. 

2. Table 1, the upper section related to Librispeech test-clean does not show the proposed system's results.

3. There are some minor issues in table captions: Tables 1 and 2 mention that (w/o SS) means without Sway Sampling, but this phrase is not used anywhere in these tables.

### Questions
1. Even though the supplemental results show WER and RTF improvements over E2-TTS, in the main text, it appears F5 performs on par with previous methods. Especially, in Table 1, the WER seems to be worse than the closely related two models. Could you please explain the comparison more clearly in the main text? 

2. Regarding the speaker similarity metrics vs. the ablation in Section 5.2: Do the authors think that Sway sampling has the advantage which is mentioned in Section 5.2 but at the same time hurting speaker similarity to some extend? 

3. Since in Table 1 the paper compares the punctuation added and capitalized version of Librispeech with the widely known original version, have the authors tried a model without including punctuation in their vocabulary? If yes, how does it compare to the presented results?

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
The authors propose F5-TTS, a zero-shot text-to-speech model that operates without explicit alignment or a pre-trained text encoder. Key distinguishing features from previous research include: (1) the use of a DiT-based structure as an estimator, (2) an additional module (ConvNeXt) for character encoding, and (3) the introduction of the Sway Sampling strategy to improve sampling efficiency. The model demonstrates respectable speaker similarity, audio quality, and pronunciation accuracy in English and Chinese benchmarks. The necessity of the proposed components is further substantiated through ablation studies.

### Strengths
- The proposed elements are clearly articulated, and each is thoroughly examined through ablation studies.

- The authors have made commendable efforts to compare their model against a wide range of baselines and benchmarks. Recently, many zero-shot TTS models, particularly those trained on large-scale datasets, have refrained from open-sourcing for various reasons, making it challenging to evaluate new models. In this regard, it is a distinct strength that the authors not only disclose their evaluation data and methods, such as LibriSpeech-PC, but also conduct comprehensive comparisons with diverse models.

- Similarly, the paper states that they plan to release their code and checkpoints soon, which is another clear advantage. Given the limited availability of large-scale models, making their code and model accessible would greatly benefit other researchers in the field.

### Weaknesses
- While I do not personally doubt the effectiveness of the proposed methodology, the contributions feel somewhat limited. This is not intended to downplay the model's design and structure for stable training. However, as the authors themselves mention, adopting a DiT structure or performing encoding to compress textual information has been already proposed in previous studies.

- Regarding Sway Sampling, a richer comparison with other sampling strategies could have highlighted its effectiveness more distinctly, especially given the recent influx of sampling techniques based on flow matching and diffusion. In addition to a thorough hyperparameter search within Sway Sampling or evaluating the effects of switching it on and off, comparisons with alternative sampling strategies would have further strengthened the impact of the proposed method.

### Questions
- I would like to inquire if comparisons with other sampling strategies from the TTS field or related domains could be provided—starting from more straightforward methods like the Heun sampler and beyond. 

- Given the lightweight approach to encoding text using an additional layer (ConvNeXt) compared to a language model, my impression is that this method may require a substantial amount of data to robustly learn pronunciation and speech-text alignment. As the authors highlight the stability of alignment training as a strength of this work, I am curious to know the lower limits of the data threshold while still maintaining stable training (although I understand that the paper's primary focus may not be on stability relative to data quantity; this question is posed out of pure curiosity). For instance, would it be feasible to achieve stable character-speech alignment training with a single-speaker dataset such as LJSpeech, which contains around 20 hours of data? Additionally, how much training time is typically needed to achieve stable pronunciation accuracy? Including results that reflect the impact of varying training data amounts would add valuable insights.

If these questions are addressed, I would like to increase the rating.

### Soundness
3

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
5

### Summary
The paper introduces F5-TTS, a non-autoregressive text-to-speech (TTS) system utilizing flow matching with Diffusion Transformer (DiT). F5-TTS uses ConvNeXt to enhance text representation and introduces a Sway Sampling strategy to boost performance and efficiency compared to E2-TTS model.

### Strengths
The paper is clearly written and easy to follow, effectively outlining its research goal of addressing the slow convergence speed and low robustness of E2-TTS, a prior work in the emerging TTS paradigm. By increasing the capacity of text representation and improving inference sampling, it presents a clear approach to tackling these issues.

### Weaknesses
We expect a scientific paper rather than a technical report. Based on this expectation, the paper lacks the following content.

Originality and Significance:

- The paper's approach to addressing the issues of slow convergence and low robustness primarily involves augmenting the model with ConvNeXt for enhanced text representation. This raises concerns about whether simply adding ConvNeXt constitutes a sufficient contribution to the TTS research community. Fundamentally, it remains unclear why E2-TTS-like architecture is capable of learning some degree of alignment without cross-attention mechanisms. Is this alignment ability a consequence of scaling up the dataset?
- Beyond presenting favorable results, the paper lacks intuitive explanations for why these improvements occur. The paper does not provide performance comparisons with other architectures that naturally emerge as alternatives to ConvNeXt, such as Transformers [1], Conformers [2], or Mamba [3].
- The proposed Sway Sampling strategy appears to be a variation of the cosine schedule, which limits its novelty. The significance of inference sampling schedules in diffusion models is already well-known and has been extensively researched in prior works like DDPM [4], DDIM [5], and DPM-Solver [6].

Quality:

- There is a significant discrepancy between the scores in the original E2-TTS paper and the reproduced one. While it is understandable that reimplementation is necessary, this raises doubts about whether E2-TTS has been accurately reproduced.
- Also, only the reimplementation's metrics are compared for E2-TTS. It would be more consistent to compare either the reported metrics from all original papers or metrics from accurately reproduced implementations. Notably, when comparing with the original E2-TTS reported metrics, F5-TTS exhibits a higher WER.
- The use of Whisper-large-v3 for measuring WER differs from the evaluation models used in other papers. This raises concerns about the fairness and comparability of the results.
- Pronunciation errors still occur in demo samples—such as failing to pronounce "y'know" in the F5-TTS sample for "Disgust" in the Emotion section of the demo. Additionally, the prosody can become completely flat with abnormal input lengths, as seen in the 1.3x Speed sample for the first prompt in the "Speed Control" section of the E2-TTS demo page. It is challenging to ascertain how much these issues are mitigated compared to the original E2-TTS.
- For the above reasons, claiming state-of-the-art performance in the conclusion may be an overstatement.
- The paper mentions that it "simply estimates the duration based on the ratio," yet there appears to be a significant deviation when ground truth is provided. Given the performance gap, it might be beneficial to include additional modules to improve duration prediction, even if it compromises some efficiency. Using duration predictors from existing methods—such as summing phoneme durations to generate the total length, directly matching the total length—could potentially enhance performance.

Clarity:

- The paper lacks clear explanations for how several hyperparameters were determined. For example, why is the CFG scale set to 2? How were the ratios selected in the two-stage control of CFG? Referring to Figure 3, it seems that values of s < −0.8 might yield better metrics.
- In line 267, the paper mentions that having some gain when providing GT implies robustness. However, robustness is typically indicated when the difference in WER between using GT and not using GT is minimal. This reasoning needs further clarification.
- The term "re-ranking" used in line 461 is not adequately explained. The content described in lines 461-464 is difficult to interpret, and its significance and connection to the overall conclusions of the paper are not clear.
- In line 241, it seems that F5-TTS still exhibits significant length differences, similar to E2-TTS.
- There are several instances where proper citations are necessary to clarify references to existing work:
    - Lines 86-87: Is the paper referring to the monotonic alignment search method from Glow-TTS?
    - Line 317: Does this line relate to the cross-sentence task introduced in Voicebox?
    - Line 319: Is the paper referencing SIM-o from Voicebox?
    - Line 431: Does this pertain to the long skip connections used in DiTTo-TTS?

[1] Vaswani, A. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*.

[2] Gulati, A., Qin, J., Chiu, C. C., Parmar, N., Zhang, Y., Yu, J., ... & Pang, R. (2020). Conformer: Convolution-augmented transformer for speech recognition. *arXiv preprint arXiv:2005.08100*.

[3] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*.

[4] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in neural information processing systems*, *33*, 6840-6851.

[5] Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *arXiv preprint arXiv:2010.02502*.

[6] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022). Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. *Advances in Neural Information Processing Systems*, *35*, 5775-5787.

### Questions
- The phrase "Fairytaler Fakes Fluent" in the title does not clearly convey the content or main contributions of the paper.
- It is worth examining whether entanglement is a structural limitation in both E2-TTS and F5-TTS. To clarify, the paper could include an ablation study where, under the same configuration and data setting as F5-TTS, a modified model uses cross-attention for text conditioning instead of adding filler-padded text to speech. This would help determine if the existing entanglement issues are reduced by changing the conditioning method.
- The paper claims that the two-stage control of CFG aids text alignment, yet it does not experimentally demonstrate this. It would be beneficial to show, through experiments, what specific alignment issues arise with the original CFG training method and how the two-stage control addresses these problems.
- Line 357: Information about the Mean Opinion Score (MOS) evaluation process is insufficient. Clarifying whether a platform like MTurk was used, the number and qualifications of evaluators hired, and providing examples of evaluation scripts would improve transparency. Additionally, mentioning studies that used similar MOS evaluation standards could provide a basis for comparison.
- Line 426: The paper refers to the performance of "F5-TTS−Conv2Text," but no related data appears in any tables or figures. The phrase "fails to learn" suggests that this model entirely failed at learning alignment, which is puzzling given its similarity to E2-TTS in setup. An explanation of why this model fails, in contrast to E2-TTS, is needed to understand the effectiveness and limitations of ConvNeXt within the model.

typo

- While the captions of Tables 1 and 2 mention results *w/o* SS, the tables themselves do not include these values.
- Line 192 should explicitly designate the filler token as <F>.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The work is an incremental engineering effort based on E2-TTS.It replaces E2-TTS's original Transformer + U-Net structure with a Diffusion Transformer. Additionally, it employs a ConvNeXt V2 to encode text sequences before passing them to the transformer. Sway Sampling strategy is introduced during model inference, which performs non-uniform sampling during inference, assigning more weight to the early stages of th steps. It can enhance both the performance and efficiency of the model.

### Strengths
The model performance is good and the audio sample is impressive. Experimental validation is sufficient and rigorous, which well demonstrates the effectiveness of each technique. Overall it's a good exploring exmrical engineering work of E2-TTS. The writing is clear and well-structured.

### Weaknesses
Although this work demonstrates impressive performance, it appears to be primarily an incremental engineering exploration of E2-TTS, rather than a novel contribution. 

The major contribution lies in replacing the transformer with DiT blocks and incorporating a ConvNeXt for text encoding. Most of the model structure settings are based on empirical ablation performance (see Section 5.1).
While the ablation study shows improvements brought by these techniques, the authors do not provide a theoretical analysis of their effectiveness, which overall makes the work contribution relatively weak. 

Furthermore, the discussion section does not offer insightful analysis.

### Questions
1. If s is getting smaller (e.g. -1, -0.9), what will be the performance of the sway sampling?

2. Although this is an empirical engineering study, I'm quite curious why ConvNeXt did not result in a performance improvement for the original E2-TTS, whereas the experiments showed that it was beneficial for DiT blocks.

3. Why Conv2audio is harmful for F5-TTS, but it is helpful for text, interms of WER?

4. Sway sampling is interesting. Have you tried to observe the mel spectrogram in different sampling stages, to verify your assumptions that "that the early flow steps are crucial for sketching the silhouette of target speech based on given prompts faithfully, the later steps focus more on formed intermediate noisy output"?

### Soundness
3

### Presentation
3

### Contribution
2
