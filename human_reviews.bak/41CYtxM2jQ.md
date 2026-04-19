# Boosting Fast and High-Quality Speech Synthesis with Linear Diffusion

- Decision: Reject
- Scores: 3, 5, 6, 3

## Abstract
Denoising diffusion probabilistic models have shown extraordinary ability on various generative tasks. However, their slow inference speed renders them impractical in speech synthesis. This paper proposes a linear diffusion model (LinDiff) based on an ordinary differential equation to simultaneously reach fast inference and high sample quality. We employs linear interpolation between the target and noise to design a diffusion sequence for training, while previously the diffusion path that links the noise and target is a curved segment. When we decrease the number of sampling steps (i.e., the number of line segments used to fit the path), the ease of fitting straight lines compared to curves allows us to generate higher quality samples from a random noise with fewer iterations. To reduce computational complexity and achieve effective global modeling of noisy speech, LinDiff employs a patch-based processing approach that partitions the input signal into small patches. The patch-wise token leverages transformer architecture for effective modeling of global information. Additionally, the model seamlessly integrates the strengths of both transformer and convolutional neural networks by utilizing a post-convolution module for fine-grained detail restoration. Adversarial training is further used to improve the sample quality with decreased sampling steps. We test this model on speech synthesis conditioned on acoustic feature (Mel-spectrograms). Experimental results verify that our model can synthesize high-quality speech even with only one diffusion step. Both subjective and objective evaluations demonstrate that our model can synthesize speech of a quality comparable to that of autoregressive models with faster synthesis speed.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a linear diffusion model (LinDiff) to synthesize waveform from mel-spectrogram, aiming to achieve fast inference speed and high sample quality. This paper contains two contributions, the first is to model the waveform based on Rectified-Flow, the second is to divide the waveform into patches, and design a model structure similar to ViT. In terms of the subjective indicator (MOS), the Vocoder proposed in this paper slightly exceeds HifiGAN and other Diffusion-based Vocoder (such as WaveGrad and FastDiff).

### Strengths
This paper contains two strengths,

First of all, the application of the structure of ViT in the waveform field is relatively novel, and this attempt should be encouraged. In theory, the model incorporates the use of a patch-wise token and the Transformer architecture for effective modeling of global information in noisy speech. This helps in capturing the contextual dependencies and improves the overall synthesis quality.

Secondly, from Table.1 in the experimental part, we can simply think that the newly proposed Vocoder has reached a new SOTA in terms of MOS.

### Weaknesses
## Weakness 1
The idea of "linear diffusion" in this paper basically comes from Rectified-Flow. The authors just apply Rectified-Flow to the audio field, and there is nothing new in machine learning theory. However, it is a pity that some works have already applied Flow-matching technology into the audio field, such as [1].

[1] Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale

## Weakness 2
The experimental part of the paper is not convincing. 

  For eg.

  HIFI-GAN V1 3.94±0.08 (MOS)

  LinDiff (1 steps) 3.99±0.06 (MOS)

  It is difficult to say that it has an advantage over the HiFiGAN model (LinDiff only gains 0.05±0.08).

HIFI-GAN V1     2.08 (MCD↓)

LinDiff (1 steps) 2.17 (MCD↓)

LinDiff is worse than HiFiGAN in terms of objective indicators.

## Weakness 3
The authors mentioned that this paper uses Transformer structure to model waveform for the first time, and the advantage of this structure is "capturing the contextual dependencies", so why is there no relevant experiment to prove the superiority of Vocoder over other Vocoders in contextual modeling?

## Weakness 4
In the current research environment, LjSpeech, a small lightweight dataset, is no longer enough to verify the superiority of the model (because everyone's scores are very high). Table 3 reveals that LinDiff performs poorly on large datasets such as Libritts.

### Questions
None

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a linear diffusion model (LinDiff) based on an ODE to simultaneously reach fast inference and high sample quality. The two main components of the Lindiff is an ODE formulation to enable linear interpolation and a Transformer based model on ground-truth (wav_gt) prediction. Experiments on LJSpeech and LibriTTS show the effectiveness of proposed method over previous baselines.

### Strengths
1. The paper propose an ordinary differential equation formulation on waveform generation, which can help model to generate relatively high-fidelity speech with limited steps.

2. The paper firstly introduce a Transformer based noise predictor for waveform generation.

3. Experiments and ablation study show that the Lindiff is better than the previous baselines.

### Weaknesses
The paper is well-written and clear. I acknowledge the contributions of the paper on ODE formulation and Transformer-based noise predictor. However, if these are the main contributions, I think more experiments should be conducted to verify the effectiveness of proposed method.

1. As for the ODE formulation, apart from the proposed formulation, there exists many other formulation (e.g., ODE in Grad-TTS/NaturalSpeech 2 and the original DDPM), which can also predict the ground-truth waveform. I think ablation on formulation (while keep the GAN and Transformer predictor be the same) is necessary to verify the contribution of proposed formulation.

2. As for the noise predictor, it is necessary to compare Transformer-based predictor with the convolutional based predictor (e.g., WaveNet based or Unet based) while keep the GAN and ODE formulation be the same to verify the effectiveness of Transformer-based predictor.

### Questions
1. The paper should give a more detailed description about patch (e.g., how to transform waveform to patch and how to transform predict path to waveform). According to the size in Section 3.4, it seems that the waveform is transformed to 256-dim STFT before formulating batch? 

2. Since patch seems to be a very sensitive parameter, it will be better if the ablation study on patch can be more detailed to show the trade-off (adding experiments of patch=16 and 32).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a linear diffusion model to reach fast inference speech and high sample quality. The authors demonstrate that its synthesis quality is on par with autoregressive vocoders while offering faster synthesis speed. They also introduce a patch-based processing approach to reduce computational complexity.

### Strengths
As far as I checked, the proposed LinDiff is technically sound. The proposed network architecture is novel. The experimental results suggest that LinDiff is capable of generating high-quality speech even with one sampling step. In the demo page, from a subjective feeling, the quality of LinDiff is better than FastDiff.

### Weaknesses
**Presentation**: It is quite hard to proceed from the section 2 (background) to the section 3 (method). I believe there are some irrelevant formulas (e.g. Eq. (4)) in section 2 that does not contribute to the design of LinDiff. These formulas might sidetrack and, to a large extent, hinder readers' understanding. A quick fix would be to cite the contents from another paper and only keep the most influential ones (e.g. Eq. (8)). Besides, I cannot find the training loss for stage 1 in Algorithm 1, please correct it for a self-contained presentation. Also, I suggest to bold all vectors and matrices, following the usual practice of ICLR papers, to differentiate them from the scalars. 

**More ablations**: I am also skeptical of the contribution of different novel points to the final performance of LinDiff, including the new architecture, three-stage training and the use of adversarial loss. Especially concerning the new architecture, I recommend the authors to compare it with some widely used architecture, e.g., UNet1D, or DiffWave. I believe objective measures such as MCD would be sufficient to confirm the superiority of the proposed architecture.

### Questions
My questions are stated above.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a linear diffusion model (LinDiff), a fast and high-fidelity speech synthesis based on conditional diffusion models with an ordinary differential equation. LinDiff incorporates Transformer and CNN architectures for effective modeling of global information and refining details. LinDiff can synthesize high-quality speech conditioned on mel-spectrograms with only one diffusion step.

### Strengths
This model uses a linear diffusion process with a flow matching training method to model speech synthesis. Experiments show that it can generate higher-quality results with fewer denoising steps. The proposed model can synthesize speech with quality comparable to the autoregressive models with faster speed.

### Weaknesses
1. The main weakness of this paper is the lack of innovation. The key point of the paper is using a linear diffusion process with flow matching; however, this has been proposed in previous work and shown to significantly reduce the number of inference steps.

2. The authors did not prove the impact of the state incorporating Transformers and CNN architectures on the results. For example, using Transformers as the backbone of diffusion is not necessarily necessary, and authors should compare it with CNN-based architectures. In fact, I don’t think using a framework like VIT is necessary for the task of vocoder. Adding self-attention to CNN architectures (e.g. WaveNet) may have similar results.

3. The experimental results did not show obvious improvement. In the case of single-step diffusion, MCD, V/UV, and F0 CORR are not as good as HIFIGAN. Note that HIFIGAN is no longer a strong baseline in the vocoder field.

4. Comparisons of objective measures of diversity deserve further discussion. I think for the task of vocoder, diversity is not an important evaluation criterion, and it may not make any difference to people's sense of hearing.

### Questions
1. The author uses discrete time steps in the process of training the model. I would like to ask the author whether he has tried using sampling continuous time.

2. Can the author provide more detailed experiments to verify the necessity of Transformers as the backbone of diffusion modeling?

3. I think this framework looks like it should serve as a general conditional speech synthesis model. However, the author only conducted experiments on the vocoder task. Can the authors verify the feasibility of the framework on more tasks (e.g., TTS)?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
