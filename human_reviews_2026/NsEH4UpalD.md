# RETA: Real-Time and Expressive Talking Head Animation without Emotion Label

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Generating photorealistic and expressive talking heads from audio faces a generative trilemma, forcing a trade-off between real-time performance, lip-sync accuracy, and emotional fidelity. We propose RETA, an end-to-end framework that resolves this trilemma. The core of RETA is a novel strategy that disentangles the audio signal into two representations. First, for robust lip-sync, we use a 3DMM as a differentiable bridge, providing strong geometric guidance within an end-to-end model to prevent error accumulation. Second, for nuanced expression, a dynamic emotion embedding is learned from audio in a completely label-free manner; this is achieved by combining cross-modal knowledge distillation from a visual expert with a novel cross-synthesis consistency loss to ensure the representation is identity-agnostic. These representations are then hierarchically injected into a single-pass GAN generator for disentangled control. RETA establishes a new state-of-the-art (SOTA) by outperforming previous methods across all key metrics, while generating high-fidelity video at speeds exceeding 55 FPS. Code will be available upon publication.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces RETA, a novel end-to-end framework for audio-driven talking head generation that resolves the generative trilemma: achieving real-time performance, accurate lip-sync, and expressive emotion. RETA does this by separating the audio signal into two distinct representations: one for robust lip-sync using a 3D Morphable Model (3DMM) as a differentiable bridge, and another for emotional expression learned directly from unlabeled audio data.

### Strengths
- The authors introduce a novel end-to-end framework, RETA, that successfully resolves the generative trilemma in audio-driven talking head generation. The method simultaneously optimizes for lip synchronization, expressive naturalness, and real-time performance without relying on external labels or explicit emotion supervision.
- The ability to generate high-fidelity video at speeds exceeding 55 FPS on a single GPU is a significant contribution, making this system suitable for real-world interactive applications.

### Weaknesses
- While label-free emotion representation learning is a significant advancement, it still relies heavily on a pre-trained visual "teacher" for emotion understanding. This dependence on a static teacher limits the model’s expressive range, as the teacher’s biases may confine it to discrete emotional categories. Adopting a more dynamic, data-driven approach could enhance expressiveness across a wider spectrum of emotions.
- Although the paper mentions RETA's real-time inference speed (55 FPS), it lacks detailed analysis of computational costs, including memory usage, training time, and energy consumption. These factors are crucial for assessing the model’s suitability for real-world applications, especially in commercial or large-scale settings.
- The results are mainly evaluated on two highly curated datasets—MEAD and HDTF—which provide strong benchmarks but do not reflect performance in diverse or uncontrolled environments. The system's effectiveness in noisy conditions or across different ethnicities remains uncertain; further testing on varied datasets would improve its generalizability and robustness.
- RETA features a complex architecture with multiple interacting components such as emotion disentanglement, hierarchical feature injection, and temporal variance masking. It is unclear which elements are most critical to its overall performance.

### Questions
- How does the model handle cases where the audio signal contains overlapping speech or background noise? Is the model robust enough to maintain accurate lip-sync and emotional expressiveness in such scenarios?
- Given the current reliance on a pre-trained visual "teacher" for emotion learning, what are the authors' thoughts on the possibility of training a more dynamic, emotion-rich model using large-scale video data without the need for external supervision?
- Can the authors provide more details on the computational resources required during both training and inference, especially for real-time applications in commercial settings? This would help in understanding the scalability of the system.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a talking head animation model that achieves real-time performance, accurate lip synchronization, and high-quality facial expressions. To this end, the authors: 

1) employ a 3DMM as an intermediate geometric prior to provide structural guidance for synchronized lip motion; 

2) use a cross-modal distillation strategy to enable the model to learn emotional cues from audio signals; 

3) design a hierarchical network structure that injects motion and emotion information in a staged manner to prevent feature interference.

### Strengths
1. The use of 3DMM as a geometric prior improves training efficiency while avoiding the noise that may arise from explicit 3D reconstruction.

2. The model achieves state-of-the-art performance in terms of inference speed, lip synchronization accuracy, and visual quality.

### Weaknesses
1. (Major) The anonymous supplementary link provided (Ln1118) was submitted after the ICLR submission deadline. When I first accessed it, the repository was empty. This raises concerns about fairness and consistency in the review process. Regarding the four provided videos:

	- All of the clips are two seconds long, which is too short to meaningfully demonstrate the model’s performance. I would expect at least one example containing a complete sentence (around six seconds or more) to better evaluate temporal quality.

	- None of the four videos include blinking behavior. Is this due to the limited expressive capacity of the 3DMM (which may not capture blinks)? This limitation could potentially weaken the expressive ability of the proposed method.
  
	- Although the authors provide a comparison with related work in Fig. 6, please include videos to substantiate the model's performance, especially comparisons with state-of-the-art GAN-based methods.


2. (Minor) The claim of “disentanglement” in Section 3.1 seems not entirely accurate. The geometric prior branch may still capture emotional cues (e.g., smiles) from the audio, while the emotion representation learning branch using the FER ViT also reflects expression dynamics correlated with speech. Therefore, the two branches may be complementary rather than fully disentangled.

3. (Minor) The concept of hierarchical injection has already become common in diffusion-based models [1, 2], where early denoising stages focus on motion and structure, middle stages handle lip and expression details, and later stages refine textures. This reduces the novelty of the proposed contribution.

References

- [1] OmniSync: Towards Universal Lip Synchronization via Diffusion Transformers, NeurIPS 2025

- [2] AlignHuman: Improving Motion and Fidelity via Timestep-Segment Preference Optimization for Audio-Driven Human Animation

### Questions
Was the Wav2Vec2 model used as the audio encoder fine-tuned during training? Two independent w2v2 models are adopted and fine-tuned in EmoTalk [3] to distentangle content and emotion features. Is a single w2v2 enough?

References

- [3] Emotalk: Speech-driven emotional disentanglement for 3d face animation

### Soundness
3

### Presentation
2

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
RETA is an end-to-end talking head animation model that produces lip-synchronized and emotionally expressive face videos in real time from speech audio. The idea is to split the audio into two parts: one driving a 3D face model for accurate mouth movement, and another yielding a learned emotion embedding that adds natural facial expressions, and feed these into a single GAN generator.

### Strengths
- The paper introduces a strategy that splits the audio input into lip-sync and expression components for generation. 3DMM is used to drive accurate lip synchronization from audio, while a separate dynamic emotion embedding captures the speaker’s expressive cues.
- The approach achieves expressive talking head animation without any explicit emotion labels. Instead, it learns a latent “emotion” representation directly from audio in a self-supervised way.
- The paper reports SOTA performance in lip-sync accuracy, visual quality, and expression realism on standard benchmarks.

### Weaknesses
- The method’s pipeline is fairly complex, relying on multiple components. It requires a pre-trained 3D face model for the lip-sync branch and a pre-trained visual emotion recognizer for distillation. If the visual expert is inaccurate/biased, it could affect the learned audio-emotion embedding.
- Because the system learns expressions implicitly from audio (and does not use explicit emotion labels or user inputs), there is limited direct control over the type or intensity of the facial expression produced.In scenarios where one might want to exaggerate or modify the emotional expression independently of the audio, RETA offers no straightforward solution.
- The paper emphasizes improved “emotional fidelity,” but it does not clearly report a rigorous quantitative evaluation of how true-to-life or perceptually correct those generated emotions are.
- It’s not fully evident how well the approach generalizes to voices or expressions outside the training data.

### Questions
- How is the “emotional fidelity” of the talking head evaluated objectively in this work? The paper claims to resolve the emotional expression aspect of the generative trilemma, but does it use any listener studies or emotion recognition metrics to verify that the generated expressions correctly convey the intended emotion?
- The model enforces that the learned audio-based expression features are identity-agnostic. Does this disentanglement ever conflict with preserving the person’s identity in the video?
- The use of a 3D morphable model provides a robust way to get lip movements, but does it constrain the range of facial motions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a method for synthesizing high fidelity audio-driven talking face animation videos in real-time (55 fps). The proposed method imposes a geometric prior in the form of 3DMM expression coefficients on a fixed identity face in 3DMM representation. Emotion representation learning via cross-modal knowledge distillation is used to learn emotion features from audio, instead of using a separate driving emotion input.  An adversarial generator is used to generate the animated face with geometric prior features injected in earlier layers and emotion features injected in later layers

### Strengths
•	The paper attempts to solve an important problem in audio-driven facial animation where lip sync, identity preservation and real-time generation can be simultaneously achieved. This is a highly challenging problem that the paper attempts to solve to a certain extent - improves the fps of state-of-the-art while not drastically reducing texture quality or emotion accuracy.
•	The paper is well organized and mostly easy to follow.

### Weaknesses
•	Lacking qualitative comparisons with emotional talking head generation methods EAT, EMMN, EVP, MEAD. In particular, what is the video fidelity and emotion expressiveness issue with EAT, EMMN that that proposed method claims to solves in the real-time setting.
•	In the supplementary video results, Lip sync accuracy seems to be poor. Much older SOA methods have better lip sync accuracy (e.g. Wav2Lip, PC-AVS[1]) than the visual results. This does not seem to justify the claim of the paper that it “simultaneously achieves accurate lip-sync, nuanced expressiveness, and real-time performance.” Also, the videos are of very short duration, which is not suitable for assessing the animation fidelity in terms of lip sync.
•	In the supplementary video results, video quality does not appear to be of very high fidelity, despite the claims of the paper. There appears no gain in texture fidelity over EchoMimic from the qualitative figures. Qualitative comparison with SOTA methods in supplementary video is desirable to judge the impact of the proposed method in realism of emotions and fidelity of video. 
•	Qualitative ablation study results are needed.
•	Missing citations :
	[1] Zhou, Hang, et al. "Pose-controllable talking face generation by implicitly modularized audio-visual representation." CVPR 2021.
	[2] Liang, Borong, et al. "Expressive talking head generation with granular audio-visual control." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
	[3] Peng, Ziqiao, et al. "Synctalk: The devil is in the synchronization for talking head synthesis." CVPR 2024.
	[4] Wang, Duomin, et al. "Progressive disentangled representation learning for fine-grained controllable talking head synthesis." ICCV 2023.
	[5] Sinha, Sanjana et al. “Emotion-controllable generalized talking face generation.”, IJCAI 2022.
	[6] Wang, Haotian, et al. "Emotivetalk: Expressive talking head generation through audio information decoupling and emotional video diffusion." CVPR  2025.

Contributions do not appear to be significant over existing work in this field.
•	Line179 states “framework disentangles the problem by mapping the audio signal to two separate and more predictable streams: geometric motion and emotion representation”  - EVP extracts disentangled content and emotion information from the input audio signal.
•	Line 14 “core of RETA is a novel strategy that disentangles the audio
•	signal into two representations”  - Disentangled Feature Learning is Section 3 has been attempted in many prior works on speech-driven facial animation, such as EVP, EAMM, PD-FGC.
•	conditional GAN generator G and image encoder E_I are borrowed from EDTalk and initialized with the pretrained weights of EDTalk.
•	Many relevant works not cited or compared (refer Weaknesses), thereby not able to establish the significance of the advancement of state-of-the-art in 2D talking head animation
•	Line 91-92 first end-to-end framework to resolve the trade-offs of prior work by simultaneously achieving accurate lip-sync, nuanced expressiveness, and real-time performance. This claim is not adequately justified in experimental results –  sync accuracy not compared with some of the related latest work (e.g PD-FGC, Wav2Lip), texture quality metrics and emotion accuracy (MEAD) do not demonstrate clear improvement over state-of-the-art. The improvement in  fps is clearly compromising image fidelity metrics, especially on MEAD dataset.
•	An ablation study is needed to justify whether use of the supervised loss on 3DMM expression coefficients degrades the lip sync accuracy, otherwise the second contribution (lines 93-94) is not properly justified.

•	Unclear phrases :
o	Line 75 - introduce a 3DMM as a differentiable bridge 
o	Line 97 - to prevent feature interference 
•	Lack of clarity in Fig 1 block b.1 : Difficult to understand the process flow from inputs f_TEX and f_STY to the Generator network
•	Complex phrase : Line 138 - autonomously distilling nuanced emotional cues directly from the input audio’s acoustic properties.
•	“primary cause of low fidelity in talking head generation is the ambiguity of direct audio-to-pixel mapping” - this statement should be rephrased to indicate lip sync quality as fidelity might imply visual quality, which is unrelated to audio.

### Questions
Please see weaknesses


•	EVP extracts emotional information from audio, it does not use emotion labels as input (unlike MEAD). According to the definition of Expressive Naturalness in lines 36-37, Why is EVP marked as not possessing expressive naturalness in Table 1.
•	Please elaborate how (Lines 186-187) 3DMM parametric representation inherently disentangles the strong audio-lip correlation from weaker correlations with head pose or blinks.
•	Why does the 3DMM "differentiable bridge" perform better than supervision on 3DMM coefficients ? 
•	How does the lip sync accuracy metric compare with Wav2Lip [1] , PD-FGC[4] and PC-AVS [3] ?
•	It is not clear which module introduces such a significant improvement in FPS over EAMM, EAT, SadTalker which are not diffusion-based.

### Soundness
2

### Presentation
3

### Contribution
2
