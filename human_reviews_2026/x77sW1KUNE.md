# T-VecTTS: Adding time-varying-emotion control to flow-matching-based TTS

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Recent advances in text-to-speech (TTS) have enabled natural speech synthesis, yet fine-grained, time-varying emotion control remains challenging. Existing methods typically provide only utterance-level control or require full-model fine-tuning with a large in-house emotional speech corpus. To overcome these shortcomings, we propose a time-varying-emotion controllable TTS (T-VecTTS), seamlessly adding the additional control to the pre-trained flow-matching-based TTS. To leverage the off-the-shelf model, we freeze the original model and attach a trainable branch that processes additional conditioning signals for emotion control. 
Moreover, we identify the flow step interval that is responsible for determining emotions and use it for detailed control.
We further provide practical recipes for emotion control on three components: (1) an optimal layer choice via block-level analysis, (2) control scale during inference, and (3) selecting the temporal emotion window size.
The advantages of our method include the zero-shot voice cloning capability, naturalness of the synthesized speech, and no need for a large emotional speech corpus or full-model fine-tuning.
T-VecTTS achieves state-of-the-art emotion similarity scores (Emo-SIM and Aro–Val SIM).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proproses a method to add fine-grained, time-varying emotion control to a pre-trained flow-matching TTS system by 1) attaching a trainable conditioning branch connected via zero-convolutions for gradually incorporate additional conditioning, and by 2) restricting control to an identified “emotion-specific flow-step” interval.

### Strengths
- This paper demonstrates that emotion is determined in the early flow steps, which is intuitive. The proposed emotion-specific flow-step concept (restricting conditioning to [0, tₑₘₒ]) is well-motivated and may have broader implications for controlling other attributes in flow or diffusion models. The reconstruction-after-perturbation experiments effectively support this observation.

- This paper presents a computationally efficient approach for introducing fine-grained, time-varying emotional control into large pre-trained TTS systems.

- This paper includes multiple ablations (flow step range, selective blocks, control scale, emotion window size) that evaluate design choices and tradeoffs (emotion vs. WER, inference time).

### Weaknesses
- The primary weakness of this work lies in the statistical validity of its experimental setup. As detailed in Appendix D, the test set appears to contain an extremely small number of samples, which raises significant concerns about the generalizability of the conclusions. This limited scale is the likely explanation for seemingly implausible results, such as the 0% WER reported in Tables 1, 2, and 5. A perfect WER is highly improbable in any realistic TTS evaluation, given the inherent recognition noise of even strong ASR models like Whisper Large-v3. **Consequently, findings based on such a small-scale testbed lack statistical significance and may not reflect the model's true performance.**

- The MOS scores in Table 6 are reported but lack confidence intervals and statistical tests.

### Questions
- Please clarify the WER = 0 entries (Tables 1/2/5). This is the most critical issue, as it makes the experimental results hard to trust.

- Please add confidence intervals and statistical tests for the MOS scores reported in Table 6.

- Please fix some presentation issues: 
  1) in line 423, typo "datset"
  2) in Table 6, there is a missing \toprule line.
  2) in Table 10, there is a missing \bottomrule line.

- ICLR appendix usually does not include a title, considering removing it.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
T-Vec TTS proposed to add time varying emotion control to pretained flow matching based TTS without degrading the base model's performance. Instead of retraining or fine-tuning a whole mode on emotional data, T-Vec TTS rather freezes a large pretrained zero shot TTS  and attaches a lightweight conditional branch to handel emotion input. This design brings several key novelties. In essence, T-VecTTS's technical innovation lies in how the emotion is integrated: by isolating when (which diffusion step) and where (which model layers) to apply the control, and doing so via a non-disuptive, additive branch. This contrasts with earlier approaches that either fine-tuned entire models (risking overfitting and quality loss) or applied controls in a coarse way

### Strengths
(1) T-VecTTS inherits zero-shot  capability from its base model. It can use a short audio prompt of any new speaker to clone that voice. Additionally, it adds emotion control on top of zero-shot TTS.

(2) No full fine tuning by freezing the original model and training only a small branch, T-VecTTS avoids the heavy cost of full model fine-tuning.

(3) Due to zero-initialized branch and selective conditioning, T-VecTTS manages to retain the high naturalness and intelligibility of the original TTS.

(4) T-VecTTS achieved state-of-the-art scores on emotion similarity metrics (Emo-SIM and Aro-Val SIM) compared to prior systems.

(5) In brief the strenghts of  T-VecTTS is to provides a practical and efficient solution: adding powerful emotion controls to a large pre-trained TTS without needing massive data or sacrificing the original model’s quality.

### Weaknesses
Following are some of my concerns 

(1) Generalization : (a) The generalization is tied to the pre-trained base model's capabilities. While it preserves the original model's naturalness and speaker identity in most cases, it can inherit the base model's weaknesses.  For example, the F5-TTS backbone struggles with extreme vocal expressions  both F5 and T-VecTTS produce artifacts for an overly excited high-pitched voice.

(b) When transferring emotion from Japanese audio into English speech, the output sometimes carries residual Japanese prosody (an accent) because the base was trained only on English/Chinese. These cases show that T-VecTTS may falter with out-of-distribution inputs (unusual pitch ranges, unseen language prosody), indicating incomplete robustness despite its zero-shot design.

(c) Since it use F5-TTS as backbone my concern is  adapting T-VecTTS to a different TTS architecture would require a similar analysis to find where to inject control without breaking fidelity. If a TTS model doesn’t have a clear emotion-specific stage, applying this approach could be non-trivial.

(2) Dependence on SER:  Although authors have higlighed this in limitation section, a core limitation of T-VecTTS lies in its dependence on an external speech emotion recognition model for quided emotions. The model only controls the emotional trajectory that the SER can detect  in case of T-VecTTS, frame-wise arousal and valence values. This means emotional expression is encoded in just two dimensions, which restricts the diversity of emotions that can be distinctly conveyed. Subtle or complex emotions that share similar arousal/valence (e.g. fear vs. anger, which are both high arousal and negative valence) might not be well distinguished, since no discrete emotion category label is explicitly used. Moreover, the reliance on SER makes T-VecTTS inherently limited by the recognizer’s accuracy and scope.

(3) Experiments: 

(a) Not all baselines were reproduced some were taken as reported in prior papers, whereas T-VecTTS was evaluated on presumably newly generated audio, which might not be perfectly comparable. 

(b) There were no preference tests or qualitative feedback on the emotional expressiveness, nor any interactive evaluations of the control interface.

(c) The paper does not explain or explote into how the time-varying emotion vector actually influences the TTS output beyond reporting improved scores. I understant that the model injects an emotion-conditioned branch at certain layers, but it’s a black box in terms of what it changes in the speech. For example, does a higher arousal primarily increase the pitch? or Does a sadness emotion elongate the speech or lower the volume? Are certain phonemes or words being pronounced differently under emotional stress?  One part of this is analyzing acoustic features: take identical text, synthesize it in different emotions or intensities, and then examine features like F0 (pitch contour), energy, duration, and spectral shape. 

(d) The selective block strategy is derived empirically to inject the emotion only at certain transformer blocks and skip others (0,1,6,16). I appreciate this that is based on clever ablation ((skipping each layer to see its effect on WER and speaker similarity) but the paper doesn’t provide insight into why those layers are the ones to exclude beyond the numerical effect. In other words, what do those critical layers do, and why would adding emotion there be detrimental? There is a theoretical gap in understanding the model’s architecture.
 
(e)  T-VecTTS uses a continuous emotion representation (via Emotion2Vec and arousal–valence values), which should, in theory, allow a broad range of emotional expressions and even mixtures. However, the paper doesn’t show any visualization of this emotion space or examples of intermediate emotions. We don’t know if the model’s “emotion space” is well-behaved (e.g., is neutral in the center and various emotions spread out appropriately?) or if it only learned the discrete emotions present in training.

### Questions
(1) I think It can generalizes well within the training distribution, but extreme voices or languages outside the base model’s training set can lead to degraded quality or accent artifacts.

(2) I wonder if there are any instances  If the SER misidentifies or smooths over an emotional nuance, the TTS will faithfully reproduce that flawed signal.

(3)  Whether the model can interpolate or smoothly transition between emotions beyond the two-scenario concatenation (EMO-Change) tested.?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes fine-tuning F5-TTS with ControlNet for time-varying emotion control in text-to-speech. The framework can also be used for emotion-preserving speech-to-speech translation. Two key techniques are highlighted: (1) identifying layers that are important for preserving TTS quality and freezing them during training, and (2) applying control only on a subset of diffusion steps.

### Strengths
The paper included many details in Appendix. Reproducibility is good since relying on open source models.

The reported results look good, achieving state-of-the-art performance in compared baselines on two benchmarks, under the constraint of using much fewer data compared to the baselines.

### Weaknesses
The proposed techniques are of limited significance. The proposed techniques are specific to the emotion control task and the F5-TTS model, so it's transferrability and significance to other TTS paradigms (such as MaskGCT [3]) are not guaranteed.

The paper focus on the small data regime (400 hours), so it should focus on this scenario and compare with other fine-tuning methods with few data. There might be simpler and more lightweight solutions, such as training a guidance model and conduct classifier guidance [1], and fintuning the model with conditional layernorm [2].

In introduction (line 77) "without modifying its original parameters" seems a bit misleading. Since training with ControlNet seems nearly double the number of parameters. Again the solution seems a bit heavy on computation cost.

Since F5-TTS does not support Japanese, the evaluations on JVNV S2ST relies on converting Japanese to Romaji. So the training inference mismatch is severe on JVNV S2ST test set. How can the NMOS and SMOS be improved with T-VecTTS fine-tuning?

For objective evaluation, please report more details as for how the results are calculate. Did you conduct repeated evaluation with different random seeds?

[1] EmoDiff: Intensity Controllable Emotional Text-to-Speech with Soft-Label Guidance
[2] AdaSpeech 4: Adaptive Text to Speech in Zero-Shot Scenarios
[3] MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer

### Questions
Please see section on weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper adds time-varying emotion control to a flow-matching TTS base model (e.g., F5-TTS) without full fine-tuning. The authors freeze the base and attach a small, trainable conditioning branch connected via zero-convolutions; the branch ingests a frame-level arousal/valence sequence from a SER model and injects it into selected DiT blocks. They also identify an emotion-specific flow-step interval and apply emotion control only in that window to reduce WER while improving emotional expressiveness.

### Strengths
- Freezes the backbone and adds a compact trainable branch tied by zero-conv, making it easy to work on strong flow-matching TTS models.
- The paper follows the EmoCtrl-TTS protocol (EMO-Change, JVNV S2ST) and reports the same metrics (WER, AutoPCP, Emo-SIM, Aro-Val SIM).
- Clear demonstration of intra-utterance transitions (e.g., sad→happy) using the reference emotion trajectory, similar with EmoCtrl-TTS
- Uses ~400 h public emotional speech and shows that emotion-rich data matters more than raw hours (LibriSpeech control).

### Weaknesses
- Limited conceptual novelty over EmoCtrl-TTS. The main novelty is how emotion is injected (a zero-conv conditioning branch + flow-step scheduling + block selection), which feels incremental relative to prior time-varying emotion control via full fine-tuning.
- Gains are strongest on emotion similarity; SIM-o/AutoPCP/WER are more mixed (and authors attribute some WER gaps to cross-lingual S2ST and base-model language coverage).

### Questions
- Why do SIM-o / WER / AutoPCP sometimes degrade? Any insights on it.
- In Figure 2, is it correct to include the ‘Noisy speech’ (training) and ‘Noise’ (inference) blocks?
- Training data: which six datasets? You mentioned "To overcome this limitation, we constructed a training dataset by combining six datasets", could you explain more regarding the details.
- A follow-up question, any studies over the training data, which one is more important to make the performance better?

### Soundness
3

### Presentation
3

### Contribution
2
