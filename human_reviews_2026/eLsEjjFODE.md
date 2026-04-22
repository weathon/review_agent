# SpeechOp: Inference-Time Task Composition for Generative Speech Processing

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 4

## Abstract
While generative Text-to-Speech (TTS) systems leverage vast "in-the-wild" data to achieve remarkable success, speech-to-speech processing tasks like enhancement face data limitations, which lead data-hungry generative approaches to distort speech content and speaker identity. To bridge this gap, we present SpeechOp, a multi-task latent diffusion model that transforms pre-trained TTS models into a universal speech processor capable of performing a wide range of speech tasks and composing them in novel ways at inference time. By adapting a pre-trained TTS model, SpeechOp inherits a rich understanding of natural speech, accelerating training and improving S2S task quality, while simultaneously enhancing core TTS performance. Finally, we introduce Implicit Task Composition (ITC), a novel pipeline where ASR-derived transcripts (e.g., from Whisper) guide SpeechOp's enhancement via our principled inference-time task composition. ITC achieves state-of-the-art content preservation by robustly combining web-scale speech understanding with SpeechOp's generative capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SpeechOp, a unified and versatile speech processor that leverages a pretrained DiT-based TTS model to perform a broad range of speech tasks (e.g., enhancement, separation, voice cloning) at inference time. The authors note that TTS models benefit from abundant in-the-wild data, S2S models typically rely on limited paired datasets, constraining their generative fidelity and data efficiency. To bridge this gap, SpeechOp employs a two-stage training paradigm, beginning with TTS pretraining followed by multi-task fine-tuning across multiple speech operations. Furthermore,  the paper proposes TC-CFG, a principled approach that fuses different modalities to improve the model performance at inference time. Experimental results demonstrate consistent improvements in both generative quality and task flexibility, validating SpeechOp as a strong step toward a universal, generative speech processor.

### Strengths
- The paper is clearly written and effectively motivates the need for unifying TTS and S2S paradigms.
- Empirical results convincingly demonstrate that initializing S2S models from a pretrained TTS backbone accelerates convergence and enhances performance.
- The proposed TC-CFG mechanism is conceptually well-founded and illustrated through a clear 1D toy example, highlighting its advantage over simple score averaging.
- The proposed framework effectively leverages different modalities to boost the task performance.
- The extensive experiments across TTS, enhancement, and separation tasks collectively validate the versatility and efficacy of SpeechOp.

### Weaknesses
(minor suggestion) Including the task embedding component in Figure 3 would further clarify how task conditioning is integrated across modalities.

### Questions
- How much additional training time does SpeechOp require compared to the standalone TTS baseline?
- Can the proposed TC-CFG mechanism be generalized to other modalities or tasks (e.g., direct application to TTS or voice conversion scenarios)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a multi-task training method that jointly learn on TTS and speech processing tasks. The paper also proposes a test time composition method using CFG to better guide the speech processing task.

### Strengths
1. While multi-tasking learning is not new, the paper propose some novel design choices that let TTS and speech processing tasks to be trained more seamlessly in a diffusion model.
2. The inference time task composition method uses text conditioning CFG to additionally guide the speech processing task and seems to have good results.
3. The paper is in general written clearly and easy to follow.

### Weaknesses
1. The competitors compared in the evaluation section are in general too few and not SoTA. For example, in TTS all models compared are from 2024. There are many new TTS models (and some good open sourced ones like Higgs Audio) in 2025. I think the evaluation section need to be updated with results comparing against more recent methods. The same for speech editing where only one work is compared against. 
2. While the paper performed a simple ablation on the task composition, it didn't run any ablation on the design choices of the training methods and model architecture.

### Questions
None

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This goal of this paper is to develop a speech model that can perform some speech to text and speech to speech tasks by starting from a diffusion based model that is trained only for TTS initially. The argument is the availability of relatively larger scale data for TTS task while lack of sufficient paired data for speech denoising, speaker separation, speech enhancement, etc. tasks. In addition to multitask training, the model allows task composition during inference time, such as Text-to-Speech + Enhancement, Text-to-Speech + Separation in one model. This is achieved by a method called the implicit task composition (ITC) where a Whisper based ASR transcripts are used in place of  text component to guide the proposed SpeechOp model's speech enhancement process. ITC does not only lead to a model capable of handling different tasks but also improves the TTS task performance that the model was initially trained on. The TTS model is based on diffusion modeling on DAC based latent speech features. In addition to text and audio features, the model hasa a learnable Task Embedding that conditions model behavior. Once the basic TTS model is trained the multitask training consists of two stages. In the first stage, TTS and S2S tasks are jointly trained. In the diffusion training step, classifier free guidance method is used. For the multitask method the training objective is a weighted sum of the speech enhancement and the TTS losses. The system is trained several open source datasets and all speech signals are upsample to 48kHz. Experiments compared the performance on both individual tasks and task composition performance. Experiments claim competitive audio quality, significantly outperforming SepFormer based speech separation  across all datasets. Especially, experiments suggest that the text guidance is an important part of the proposed SpeechOp.

### Strengths
Originality
+ The paper proposes a text guided multitask modeling for several speech tasks in one model that was initially trained for TTS. Thus the model can leverage the relatively high-resource TTS task in other speech related tasks such as speech separation, and inclusion of the text guidance allow for inference time task composition that can expand the model capabilities. 

Quality 
+ Experimental results suggest that the generated audio quality is competitive as compared to audio reconstruction with a HiFiGAN2 vocoder. In speech separation, the paper outperforms the Sepformer based baselines.  

Clarity 
+ Mostly clearly written. 
+ Datasets used in the experiments are open source. Some experimental settings are provided in the appendix. 
  
Significance
+ Since the SpeechOp paper is a multitask speech model capable of handling several speech tasks, it is relevant to various subsets of speech researchers.

### Weaknesses
The following points are mostly about the clarity of the content or the quality of the experimentation. 

- One point is that is not very clear is how critical is the task embedding in the system if Fig. 3. or whether given the complex interactions between diffusion latents and text encodings through cross attention, it is not clear whether the model can leverage the task embedding efficiently.

- In some experiments, SpeechOp results are presented in (no transcript) and (ground truth transcript) versions. For completeness, it could have been useful to see the SpeechOp results with the ASR transcriptions (e.g. in Table 4).

- In Table 3, there is a reference to Speaker Personalization but I could not see the explanation of how this is achieved. Could you please explain?

### Questions
1. Please discuss the effectiveness of the task embeddings in the SpeechOp model more clearly. 

2. In some experiments, SpeechOp results are presented in (no transcript) and (ground truth transcript) versions. For completeness, it could have been useful to see the SpeechOp results with the ASR transcriptions (e.g. in Table 4). Could the authors please comment on that?

3. In Table 3, there is a reference to Speaker Personalization but I could not see the explanation of how this is achieved. Could you please explain?

4. (Minor typo) Section 5. Speech-to-Speech Pathway: us a complex alignment mechanism -> *use* a complex alignment mechanism

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
This paper presents SpeechOp, a multi-task latent diffusion model designed to transform pre-trained Text-to-Speech (TTS) models into a universal speech processor capable of performing and composing multiple speech-to-speech (S2S) tasks (e.g., enhancement, separation, voice cloning) at inference time. The key innovation lies in the Task Composition Classifier-Free Guidance (TC-CFG) mechanism and Implicit Task Composition (ITC) framework, which allow flexible inference-time combination of speech operations using text or ASR-derived guidance (e.g., from Whisper). The authors show that TTS pre-training accelerates convergence and improves downstream S2S performance, and that SpeechOp achieves state-of-the-art content preservation across tasks such as enhancement and separation.

### Strengths
1. The idea of repurposing pre-trained TTS models for versatile S2S operations via inference-time task composition is well-motivated, addressing the data scarcity challenge in generative S2S tasks.
2. The paper reports substantial improvements in speech enhancement and perceptual quality gains in speaker separation.

### Weaknesses
1. The idea of leveraging pre-trained diffusion-based TTS models as clean speech generators for downstream tasks such as denoising, enhancement, and separation has already been proposed in Voicebox, Audiobox, and Speechflow. The paper’s novelty mainly lies in its composition mechanism, rather than the overall concept of using TTS priors for multi-task speech processing.
2. While the composition strategy is interesting, the underlying model (latent diffusion with DiT backbone) largely follows existing designs (Voicebox, DiTTo-TTS, etc.). The architectural contribution is modest.
3. ITC’s success depends heavily on accurate ASR transcripts. The paper lacks robustness analysis under ASR noise or low-resource conditions. Maybe using some softer semantic supervision like SSL features instead of the explicit transcripts would mitigrate this issue.

### Questions
How does SpeechOp perform on cross-lingual or non-English speech tasks, given the ByT5 encoder and English-only datasets?

### Soundness
3

### Presentation
3

### Contribution
2
