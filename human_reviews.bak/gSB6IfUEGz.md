# VoxGenesis: Unsupervised Discovery of Latent Speaker Manifold for Speech Synthesis

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 1

## Abstract
Achieving nuanced and accurate emulation of human voice has been a longstanding goal in artificial intelligence. Although significant progress has been made in recent years, the mainstream of speech synthesis models still relies on supervised speaker modeling and explicit reference utterances. However, there are many aspects of human voice, such as emotion, intonation, and speaking style, for which it is hard to obtain accurate labels.
In this paper, we propose VoxGenesis, a novel unsupervised speech synthesis framework that can discover a latent speaker manifold and meaningful voice editing directions without supervision. VoxGenesis is conceptually simple. Instead of mapping speech features to waveforms deterministically, VoxGenesis transforms a Gaussian distribution into speech distributions conditioned and aligned by semantic tokens. This forces the model to learn a speaker distribution disentangled from the semantic content.
During the inference, sampling from the Gaussian distribution enables the creation of novel speakers with distinct characteristics. More importantly, the exploration of latent space uncovers human-interpretable directions associated with specific speaker characteristics such as gender attributes, pitch, tone, and emotion, allowing for voice editing by manipulating the latent codes along these identified directions.
We conduct extensive experiments to evaluate the proposed VoxGenesis using both subjective and objective metrics, finding that it produces significantly more diverse and realistic speakers with distinct characteristics than the previous approaches. We also show that  latent space manipulation produces consistent and human-identifiable effects that are not detrimental to the speech quality, which was not possible with previous approaches.
Finally, we demonstrate that VoxGenesis can also be used in voice conversion and multi-speaker TTS, outperforming the state-of-the-art approaches. Audio samples of VoxGenesis can be found at: \url{https://bit.ly/VoxGenesis}.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a speaker creation method that uses a pre-trained speaker embedding model together with Gaussian sampling to generate new speakers.

### Strengths
1. The proposed model performs better than Tacospawn in generating unseen and diverse speakers.
2. Ablation studies are performed to understand the impact of different design choices.
3. The paper is in general written clearly with a good amount of illustrations and maths.

### Weaknesses
1. I think the novelty is somewhat limited. To me, the proposed method is quite similar to a VAE-based speaker encoder but with the encoder changed to a pre-trained speaker embedding module. The mapping network is almost the same as in StyleGAN and is quite similar in principle to AdaSpeech. My main takeaway is that a pre-trained speaker embedding model as the encoder of a VAE works well for generating unseen speakers, and NFA is a promising speaker embedding model compared to other pre-trained speaker embedding model.
2. The baselines compared are quite old by deep learning standards. There are a few follow-up works to TacoSpawn. For zero-shot VC there are many more recent works (e.g. ControlVC). The authors state VITS is the "the state of the art" Multi-Speaker TTS but it is by no means true in 2023.

### Questions
For Table 1, it is not clear to me for each baseline, what is the exact procedure and hyper-params involved to sample speaker embeddings. I hope the authors can clarify.

### Soundness
3 good

### Presentation
3 good

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
This work proposes a neural factor analysis (NFA)-based speech synthesis model for tts, vc, and voice generation. They simply disentangle a HuBERT representation for controlling the semantic guided voice representations.

### Strengths
They utilize a neural factor analysis (NFA) for speech representation disentanglement, and adapt the semantic representation with style conditions. This simple modification could improve the TTS and VC performance by adopting it to speech resynthesis frameworks.

### Weaknesses
1.	The authors should have conducted more comparisons to evaluate the model performance. They only compare it with speech resynthesis, and the result is a little incremental.

2.	They utilize a NFA which was proposed in ICML 2023. The contribution is weak.

3.	It would be better if you could compare the self-supervised speech representation model for the robustness of your methods by replacing HuBert with any other SSL models.

4.	Using HuBERT representation may induce a high WER. Replacing it with ContentVec may improve the pronunciation.

5.	The semantic conditioned Transformation is utilized in many works. NANSY++ utilizes a time-varying timbre embedding and this is almost the same with this part.

### Questions
1.	The authors cited LibriTTS and LibriTTS-R together. Which dataset do you use to train the models? In my personal experience, using LibriTTS-R decreases the sample diversity. Do you have an experience with this?

2.	According to ICLR policy, when using human subjects such as MOS, you may include the evaluation details in your paper.

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
This paper proposes an unsupervised voice generation model called VoxGenesis by transforming Gaussian distribution to speech distribution. The proposed VoxGenesis can discover a latent speaker manifold and meaningful voice editing directions without supervision, and the latent space uncovers human-interpretable directions associated with specific speaker characteristics such as gender attributes, pitch, tone, and emotion, allowing for voice editing by manipulating the latent codes along these identified directions.

### Strengths
1. It is interesting to utilize Gaussian distribution transformation for unsupervised voice (speech) synthesis so that the model is able to generate realistic speakers with distinct characteristics like pitch, tone, and emotion.

### Weaknesses
1. While the idea is promising, the experimental results seem to be limited. Most of the performances are from the ablation studies. The proposed model should compare the performances with the previous works like SLMGAN [1], StyleTTS [2], and LVC-VC [3]. Moreover, the paper only utilizes one dataset, LibriTTS-R. More extensive experiments on different dataset might be necessary.
2. The paper can be more curated. While it is well written paper, it slightly lacks in structure. Since the idea is interesting enough, I would consider adjusting the rating if the paper is more well structured and the additional experiments are conducted.

[1] Li, Yinghao Aaron, Cong Han, and Nima Mesgarani. "SLMGAN: Exploiting Speech Language Model Representations for Unsupervised Zero-Shot Voice Conversion in GANs." 2023 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2023.

[2] Li, Yinghao Aaron, Cong Han, and Nima Mesgarani. "Styletts: A style-based generative model for natural and diverse text-to-speech synthesis." arXiv preprint arXiv:2205.15439 (2022).

[3] Kang, Wonjune, Mark Hasegawa-Johnson, and Deb Roy. "End-to-End Zero-Shot Voice Conversion with Location-Variable Convolutions." Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH. Vol. 2023. 2023.

### Questions
Please refer to the weakness section.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper discusses on an approach on modeling speaker's voice in speech generation models, as well as its application in voice conversion and multispeaker TTS.

### Strengths
Not clear to me.

### Weaknesses
* The clarity of the presentation and writing needs improvement. Overall, it's difficult to read this paper. Some basic writing styles, like missing necessary parentheses on citations, is very bothersome for reading. The description of the method seems over complicated than what the method actually is.

* A lot of technical incorrectness. Just a few examples:
   - Sec 3.1: "GAN has been the de facto choice for vocoders." This is a false claim and ignores a large arrays of active and important works in the community. There are other popular vocoders choices like WaveNet, WaveRnn, diffusion-based approaches etc.
   - Sec 3.1: "A notable limitation in these models is ... learn to replicate the voices of the training or reference speakers rather than creating new voices." Another false claim. It improper to state for such an "limitation" because that is not the goal of the task of vocoders.
   - Sec 3.1 "consequently, a conditional GAN is employed to transform a Gaussian distribution, rather than mapping speech features to waveforms". This is another improper comparison as what GAN does is to transfer the conditioning features (e.g. speech features) , rather than plain Gaussian noise, to waveforms.
   - Sec 3.1 "It is crucial, in the absence of Mel-spectrogram loss, for the discriminator to receive semantic tokens; otherwise, the generator could deceive the discriminator with intelligible speech." I don't see the connection.
   - Sec 4.2 says speaker similarity is evaluated in a 3-point scale from 0 to 1, but Table 3 shows speaker similarity as 4.x and 3.x values.

### Questions
None.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor
