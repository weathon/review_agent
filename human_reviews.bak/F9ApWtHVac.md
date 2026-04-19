# Bridge-TTS: Text-to-Speech Synthesis with Schrodinger Bridge

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
In text-to-speech (TTS) synthesis, diffusion models have achieved promising generation quality. However, with the pre-defined data-to-noise diffusion process, their prior distribution is restricted to a noisy representation, which provides little information of the generation target. In this work, we present a novel TTS system, Bridge-TTS, making the first attempt to substitute the noisy Gaussian prior in established diffusion-based TTS methods with a clean and deterministic one, which provides strong structural information of the target. Specifically, we leverage the latent representation obtained from text input as our prior, and build a fully tractable Schrodinger bridge (SB) between it and the ground-truth mel-spectrogram, leading to a faster generation process. Moreover, the tractability and flexibility of our proposed SB formulation allow us to empirically study the noise schedule and the model parameterization in training, as well as developing training-free stochastic and deterministic samplers with theory-grounded analyses of the bridge SDE and ODE, which further enrich our design spaces for exploring better generation performance. Experimental results on the LJ-Speech dataset illustrate the effectiveness of our method in terms of synthesis quality and sampling efficiency, outperforming the diffusion counterpart Grad-TTS in 50-step synthesis and strong fast TTS models in few-step scenario.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents Bridge-TTS, which incorporates the Schrödinger Bridge concept into text-to-melspectrogram generation. By introducing the Schrödinger bridge that directly connects the deterministic latent representation from the text encoder and the data, it allows for more direct use of the text encoder output compared to Grad-TTS. It demonstrated better sample quality with fewer sampling steps than Grad-TTS on the LJSpeech dataset.

### Strengths
1. This paper applies a strong theoretical background on the Schrödinger Bridge and the recent sampling methods from diffusion models to TTS.

2. Bridge-TTS shows better sample quality with fewer sampling steps compared to Grad-TTS, and even with very few sampling steps (2-step generation), it produces reasonably good quality samples.

### Weaknesses
1. The performance improvement compared to Grad-TTS is marginal. Grad-TTS is a paper published in ICML 2021, and improving upon this baseline for a single speaker dataset (LJSpeech) doesn't seem to be a challenging issue in speech synthesis at present and appears to be a straightforward application. Thus, I believe it would be difficult for it to be published in ICLR 2024. Exploring this new generative model seems to have a fresh aspect, so targeting a speech-related venue might be more appropriate. Personally, I feel that the TTS problem for single speaker datasets in the years 2023-2024 is relatively a toy problem. Generating high-quality samples from LJSpeech doesn't appear to be a challenging issue, and I'm not particularly motivated by applying the Schrödinger bridge to TTS given the experimental results in the paper.

### Questions
* The paper explores the scalar values of f and g, and shows the CMOS ablation results in Table 4. If the mel-spectrogram data is normalized to have values between [-1, 1] before training, couldn't we simply use 0 for f and 1 for g? By using f=0 and some scaling value for g, this approach appears to be the application of the Brownian bridge by Tong et al to TTS, as also mentioned in the paper.

* The Schrödinger bridge is used in the image domain for applications like Image-to-image translation. If the Schrödinger bridge offers advantages in TTS not only for sampling speed on a single speaker but also for applications like translation in the speech domain (e.g., speech-to-speech translation, voice conversion, etc.), highlighting such results would have provided stronger motivation for its application. This could have made the research more compelling.

### Soundness
3 good

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
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents Bridge-TTS that generates mel-spectrograms from deterministic text latent representations. To achieve this, the authors first introduce a fully tractable Schrodinger bridge for paired data in TTS modeling. Subsequently, they propose a novel first-order discretization of the Bridge SDE/ODE for accelerated sampling. Experimental results emphasize that the proposed approach offers synthesis quality comparable to or surpassing baseline methods, especially in few-step sampling scenarios.

### Strengths
* The authors introduce a theoretically novel approach to employ Schrodinger bridge for TTS that produces outputs from deterministic latent representations.
* They present a new sampling scheme optimized for fast sampling.

### Weaknesses
* A lack of empirical validation for the superiority of the proposed method.

It is uncertain whether the proposed deterministic latent representations are superior to the noisy Gaussian conditional prior distribution of diffusion-based TTS. Experimental results suggest that for fewer than 50 sampling steps, the proposed method seems to yield better sample quality compared to other models. However, at 1000 steps, the diffusion-based TTS model, namely Grad-TTS, outperforms the proposed methodology.

Accordingly, for a fair comparison concerning its efficient sampling scheme, it would be appropriate to contrast it with diffusion-based TTS models using a sampling scheme like the DDIM.

### Questions
It would be essential for the authors to provide an explanation for the observed worse sample quality of their proposed method compared to the baseline diffusion-based approach at 1000 sampling steps.
Additionally, the generation quality between the proposed method and diffusion-based methods using a comparable sampling scheme in few-step sampling scenarios would also be an interesting aspect of this research.

typo: (p.7, Sec.4.1.,) English graphme -> English grapheme

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a novel neural TTS system based on SB. Informative prior in generative models is an important technical point that is well handled by this paper. In certain scenarios, they show improvement over SOTA methods like Grad-TTS and DiffSinger. The proposed tractable SB is created by defining a reference SDE in alignment with diffusion models. Bridge sampling is discussed in this context to generate the target when trained with paired data (clean text, mel spectrogram). Real-time Factor (RTF) is also discussed alongside MOS, CMOS in evaluation on LJSpeech. Proposal shows improvement in reducing high-frequency artifacts.

### Strengths
1. Novelty: Introducing SB to TTS domain.
2. Technical rigor
3. Ablation studies

### Weaknesses
1. Generalizability of the proposed method is an issue. We don't know if this works on other test sets, other speakers, etc. I am not confident if this model is vast improvement in the TTS research space.
2. Focus of paper seems to be on technique. TTS-related discussion is lesser than expected.
3. "NFE" is not defined. For new audience, it could be an issue.
4. References are needed to say that some improvement in MOS is actually significant (which is what authors are conveying).
5. Some more intuition on SB would be nice for audience with less background knowledge. Maybe a graph of training with loss or some term going down.

### Questions
1. I would like to know if pre-trained models (text encoders) can be leveraged to fasten or improve the proposed SB solution.
2. Would authors like to comment on phoneme-level improvement or provide some information on trends?
3. Any word on if there is a trade-off between the quality of the synthesized speech and the computational efficiency of the method?
4. Can you add some additional consistency loss term which can further brings artifacts down?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a non-autoregressive TTS model called Bridge-TTS, which is build on the Schrodinger bridge (SB). Bridge-TTS follows the two-stage TTS pipeline, i.e., the TTS system comprises with a text-to-spectrogram acoustic model and spectrogram-to-wave vocoder model, where the Schrodinger bridge is used in modeling the former. Unlike most diffusion-based TTS models, Bridge-TTS uses deterministic prior, which is learned from the text input in a deterministic way via a text encoder module. Bridge-TTS is able to use diffusion-TTS-like sampling procedure to synthesis samples from the prior, where different SDE/ODE-based samplers can be adopted to trade-off the inference speed and the sample quality. In general, this submission makes a clear presentation, making detailed and well-structured explanations from the theories of diffusion models to that of the Schrodinger bridge, and decent derivation of the methodology in adopting SB in paired data modeling, e.g., TTS task, as long as the training objective and different sampling schemes. A singer-speaker TTS experiment using the well-known TTS benchmark corpus LJ-Speech is conducted to verify the arguments by the paper.

### Strengths
- This paper introduces yet another generative model, i.e., the Schrodinger bridge, for tackling the TTS task.
- This paper presents derivations for bridge sampling in the context that the number of sampling steps is small for the first time, and gives exact solution and 1st-order discretization of SB SDE and ODE, allowing for efficient sampling with SB-based generative models. Relationship of the solution to some famous sampling schemes, such as DDIM, is also presented.
- This submission has source codes, which could be helpful for reproducing the results presented.

### Weaknesses
Novelty:

The methodology of Bridge-TTS introduced in this submission does not attempt to address the most urgent issues of in the TTS research field, e.g., the prosody modeling, and the paper doesn’t even specify which duration modeling and text-spectrogram alignment scheme are employed. Moreover, the contribution is incremental since this paper introduces yet another kind of generative model into TTS and does not receive significant performance improvement according the experimental results presented. This submission argues that  replacing the noisy prior in previous systems with the clean and deterministic prior can boost the TTS sample quality and inference speed. However, similar arguments have been made and verified in previous works, such as DiffSinger and DiffGAN-TTS. If we look at the training scheme and loss objective carefully, the text encoder output $z$ is in fact the coarse predicted Mel spectrogram as in DiffSinger, which is learned by using the simple MSE-based reconstruction loss. The SB-based module is indeed a spectrogram post-processing module or a “spectrogram super-resolution module”, and can only refine the details of the produced spectrogram and can not fully leverage the generative modeling power of diffusion-based or SB-based models. In this regard,  the contribution of this paper is not sufficient and only incremental.

Experiments:

- Only conducted single-speaker TTS on the LJ-Speech corpus: this is not sufficient since TTS models have reached human-level quality on this data, e.g., NaturalSpeech and StyleTTS-2. It will be more sound if multi-speaker or even multi-emotion TTS experiments are conducted.
- The reason for explaining why Grad-TTS with 1000 NFEs has higher MOS score than that of Bridge-TTS with 1000 NFEs is not convincing.

### Questions
- Why the MOS scores of “Recording” and “GT-Mel-Voc.” in Table 2 are so different from those in Table 3?
- Why Grad-TTS (NFE=1000) is faster than Bridge-TTS (NFE=1000) in terms of RTF, and in other cases such as NFE=50 and 4, Grad-TTS is slower than or has equal RTF to Bridge-TTS?
- Why do you think SB-based spectrogram post-processor is better than diffusion-based ones, e.g., coarse predicted Mel spectrogram as condition to Grad-TTS decoder? Is there an intuitive explanation?
- How do you align text and spectrogram during training and how do you model phoneme durations?

Typos and minor edits in presentation:

- There is no definition of $\Psi$ and $\hat\Psi$.
- I think the symbols of “forward score” and “backward score” in the title of Table 1 are reversed.
- “In practice, we prefer the noise prediction” → “In practice, we prefer to the noise prediction”

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
