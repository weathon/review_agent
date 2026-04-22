# ApoAvatar: Expressive Audio-Driven Avatar Generation via Refocused Audio-Pose Priors

- Avg Score: 5.60
- Decision: Reject
- Scores: 4, 8, 6, 4, 6

## Abstract
Audio-driven human video generation has greatly improved lip synchronization. However, most methods still use audio mainly to control the mouth, while the relationship between speech rhythm and body motion remains weak. This often makes generated characters look unnatural. We present \textbf{ApoAvatar}, a diffusion-based framework that ties speaking style to motion dynamics. We introduce an Audio–Pose Prior Refocusing mechanism, which adjusts pose guidance based on audio intensity. Strong accents increase gesture magnitude, while quiet parts suppress unnecessary motion. We also design a frame-wise audio–video interaction module. It updates audio features using the current visual context and the refocused pose prior through a designed bidirectional cross-attention. This yields better short-term synchronization and motion coherence. The framework supports both pose-controlled and pose-free inference within one model. Extensive experiments on EMTD and HDTF show clear gains over strong baselines in lip–audio synchronization, gesture expressiveness, and overall motion naturalness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes ApoAvatar, a DiT-based framework for audio-driven video generation, aiming to address the common issue in existing methods where body motions are poorly synchronized with speech rhythm, resulting in stiff and unnatural animations. ApoAvatar introduces two key innovations: 1) an Audio-Pose Prior Refocusing mechanism that dynamically modulates the strength of pose guidance based on frame-level audio intensity—strong accents amplify gesture magnitude, while quiet segments suppress unnecessary motion, thus aligning gesture dynamics with speaking style. 2) a Frame-Wise Audio–Video Interaction module that employs a bidirectional cross-attention within an Audio DiT adapter, enabling audio features to be refined using the current visual context and the refocused pose prior, producing "pose-aware" audio embeddings. The framework supports unified inference with or without pose input. Experiments on the EMTD and HDTF datasets demonstrate that ApoAvatar outperforms baselines in lip-audio synchronization, gesture expressiveness, motion naturalness, and identity preservation.

### Strengths
* This paper identifies a critical yet often overlooked limitation in most existing works: audio and pose features are typically modeled as independently contributing modalities with insufficient interaction. To address this issue, the authors propose well-designed solutions, whose effectiveness is thoroughly validated through ablation studies. This work offers the community a novel and valuable perspective for optimizing audio-driven video generation.
* This work introduces an audio-aware pose prior refocusing mechanism that acts as a pose feature refiner. By dynamically modulating the retention level of pose embeddings based on prosodic cues, the method naturally scales gesture amplitude according to speech intonation. It amplifies movements during accented syllables and suppresses redundant motions during silent segments, thereby generating more expressive and rhythmically coherent full-body animations.
* Similarly, the paper designs a frame-wise audio-video interaction architecture that serves as an audio feature refiner. At each denoising step, the model updates audio representations by conditioning on both the current video state and the refocused pose prior, transforming audio features from static inputs into dynamically evolving, "video-aware" signals. This significantly enhances short-term synchronization and motion smoothness.

### Weaknesses
* The experimental setup is insufficiently described: Section 4.1 lacks essential details regarding the experimental configuration, including—but not limited to—the number of training stages, and whether full-parameter fine-tuning, LoRA, or SFT was used for parameter updates.
* Insufficient discussion on the text modality: The backbone model used in this work is an image-to-video (I2V) model, in which the text input inherently provides strong conditioning. However, the paper does not adequately discuss the role of the text branch, such as how the text labels in the training data are obtained or what specific text inputs are used during comparison with other methods. Notably, methods like MultiTalk[1] and OmniAvatar[2] do not use pose conditioning; hence, when text descriptions are insufficient, their generated body motions may naturally be less expressive. This oversight may unintentionally give the proposed method an unfair advantage in comparative evaluation.
* Qualitative comparison is inadequate: The paper only provides nine qualitative cases in the appendix, which do not even include the example shown in Figure 4. Moreover, while OmniAvatar[2] is included in quantitative comparisons, it is missing from the qualitative results. These issues raise concerns that the presented results may be cherry-picked, failing to fully demonstrate the effectiveness and superiority of the proposed approach.
* Concerns regarding reproducibility: The work builds upon Goku[3], a foundational model that is neither open-sourced nor commercially available. Despite initial promises by the Goku[3] authors to release it, the model remains inaccessible to the public as of now. This severely undermines the reproducibility of the work, as readers cannot verify whether the generated results stem from the base model's capabilities or from the contributions introduced in this paper. The experimental results would be significantly more convincing if the proposed method were implemented and evaluated on an open-source foundation model such as Wan[4], which is already used by several compared methods (e.g., FantasyTalking[5], MultiTalk[1], InfiniteTalk[6], OmniAvatar[2]).

[1]Kong, Zhe, et al. "Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation." arXiv preprint arXiv:2505.22647 (2025).

[2]Gan, Qijun, et al. "OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation." arXiv preprint arXiv:2506.18866 (2025).

[3]Chen, Shoufa, et al. "Goku: Flow based video generative foundation models." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[4]Wan, Team, et al. "Wan: Open and advanced large-scale video generative models." arXiv preprint arXiv:2503.20314 (2025).

[5]Wang, Mengchao, et al. "Fantasytalking: Realistic talking portrait generation via coherent motion synthesis." arXiv preprint arXiv:2504.04842 (2025).

[6]Yang, Shaoshu, et al. "InfiniteTalk: Audio-driven Video Generation for Sparse-Frame Video Dubbing." arXiv preprint arXiv:2508.14033 (2025).

### Questions
* Regarding the training procedure: Could the authors clarify the overall training pipeline and detailed training configurations? Specifically, how are the text labels in the training data obtained, and how is the influence of text conditioning on pose generation balanced?
* Regarding comparisons: Are the text inputs kept consistent across all compared methods? Additionally, could the authors provide more generation results via an anonymous link to better support the claims in the paper?
* Role of pose conditioning: When pose guidance is available, how does the proposed method compare in performance against recent pose-driven approaches such as RealisDance-DiT[1] and X-UniMotion[2]?
* Evolution of the pose branch: Omni-Human1.5[3] deliberately removed pose conditioning in its upgrade from Omni-Human1[4], as challenges in body motion generation—such as body turning, finger articulation, and dance movements—remain difficult. In contrast, this work weakens the role of the text branch and reintroduces the pose branch. What deeper insights or design considerations motivate this architectural choice?

[1]Zhou, Jingkai, et al. "RealisDance-DiT: Simple yet Strong Baseline towards Controllable Character Animation in the Wild." arXiv preprint arXiv:2504.14977 (2025).

[2]Song, Guoxian, et al. "X-UniMotion: Animating Human Images with Expressive, Unified and Identity-Agnostic Motion Latents." arXiv preprint arXiv:2508.09383 (2025).

[3]Jiang, Jianwen, et al. "Omnihuman-1.5: Instilling an active mind in avatars via cognitive simulation." arXiv preprint arXiv:2508.19209 (2025).

[4]Lin, Gaojie, et al. "Omnihuman-1: Rethinking the scaling-up of one-stage conditioned human animation models." arXiv preprint arXiv:2502.01061 (2025).

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
To address the issue that avatar poses in current talking head generation tasks are overly rigid and lack vividness, this paper proposes an "Audio-Pose Prior Refocusing Mechanism" and a "Frame-Wise Audio–Video Interaction Strategy" to enhance the modeling between audio and motion, thereby achieving more vivid avatar generation results. The paper demonstrates the effectiveness of the proposed scheme through extensive quantitative and qualitative experiments as well as user studies.

### Strengths
- The paper has a clear research motivation and well-organized presentation. Additionally, it conducts extensive experiments and compares with numerous recent baselines, which is highly convincing, demonstrating sufficient and solid research efforts.
- Rigid movements in digital human driving are a practical problem, and the paper proposes an effective solution to tackle this issue.
- The paper presents a highly general talking head generation framework that supports full-body driving. Compared with recent methods focusing only on the head, this framework holds greater practical significance.

### Weaknesses
- The paper points out the lack of interaction between pose and audio, stating that "audio, pose, and visual context are often merged by simple concatenation or a single round of attention." This issue seems to target single-stream DiT. Does this problem still exist for dual-stream (or multi-stream) MM-DiT? Since dual-stream/multi-stream designs inherently introduce interactions between modalities.
- Using accents as the prior for pose is reasonable; however, this approach of leveraging inductive bias may make it difficult to learn certain modes that do not rely on this prior.

### Questions
Please refer to the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the challenge of generating expressive audio-driven human avatars by improving audio-motion coupling. The authors propose ApoAvatar, a diffusion-based framework featuring an Audio-Pose Prior Refocusing mechanism that adjusts pose embeddings based on frame-level audio intensity. A Frame-Wise Audio-Video Interaction module refines audio features using visual context and refocused pose priors via bidirectional cross-attention. The model supports both pose-conditioned and pose-free inference. Evaluations on EMTD and HDTF datasets demonstrate improvements in lip synchronization, gesture expressiveness, and motion naturalness.

### Strengths
1. Novel audio-prosody modulation: The Audio-Pose Prior Refocusing dynamically scales gesture intensity using RMS-based loudness, addressing audio-motion decoupling.

2. Effective multimodal interaction: Frame-Wise Cross-Attention enables audio refinement via video/pose context, improving short-term synchronization.

3. Flexible inference design: Decoupled classifier-free guidance allows per-modality control, and pose-free inference maintains robustness.

4. Rigorous evaluation: Comprehensive metrics and user studies validate superiority in lip sync and motion naturalness.

5. Strong ablation support: Tables 4-5 isolate contributions of refocusing and interaction modules.

### Weaknesses
1. Superficial hand motion analysis: Hand stability is evaluated only via HKC/HKV, lacking perceptual metrics or qualitative examples of failure cases.

2. Incomplete baseline comparison: OmniHuman’s exclusion from main tables limits quantitative context, despite its relevance as a state-of-the-art method.

3. Underexplored failure modes: Occlusion robustness is mentioned in Appendix Fig. 7 but not quantitatively analyzed or discussed in the main text.

4. Limited dataset diversity: Training uses curated single-person clips; generalization to complex scenes, such as multi-person interactions, is unverified.

5. Ambiguous audio alignment details: The hop length h and sampling rate sr for RMS calculation are unspecified, hindering reproducibility.

### Questions
1. Can you include OmniHuman’s quantitative results (e.g., FID/SYNC) on a subset of EMTD/HDTF compatible with its 15-second input limit? Because OmniHuman is a key SOTA comparator (Sec 4.2). Partial results would contextualize claims of superiority more fairly.

2. For pose-free inference under occlusion in Appendix Fig. 7, report metrics of FID and FSIM on occluded vs. clean samples. How does refocusing mitigate pose errors? The claim of robustness lacks quantitative support.

3. You state pose is "heavily dropped during training" in Sec 3.4. What was the dropout rate? If pose is dropped >40% of frames, how does this impact gesture diversity??Because High dropout may explain limited motion variety in baselines as shown in Fig. 5.

4. The author mentions that the amplitude of the motion is coupled with the intensity of the audio. However, the types of motions are essentially limited to a single vertical hand gesture. In comparison, OmniHuman-1.5 possesses a greater variety of hand gestures, offering more diversity. It remains unclear whether the gesture motions can be generalised to encompass a wider range.

5. The author claims mentions that the model can ensure head-stable positioning, but is this truly an aspect requiring improvement? Appropriate head rotation should render video subjects appear more natural. Might the article be solely focused on hand poses while overlooking the coupling between the head, gaze, and audio content?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel framework for audio-driven human video generation. Specifically, it introduces an Audio-Pose Prior Refocusing mechanism to enhance the coupling between audio and motion, and a Frame-Wise Audio–Video Interaction module that refines audio features by leveraging both the visual context and the refocused pose prior. Extensive experiments demonstrate the effectiveness of the proposed approach, showing promising results.

### Strengths
1. The Audio-Pose Prior Refocusing mechanism is novel and interesting.

2. The paper’s presentation is clear and well-structured.

3. Comparisons with prior work are fair, thorough, and self-contained.

### Weaknesses
1. The Frame-Wise Audio–Video Interaction may accumulate errors if the refocused pose prior is inaccurate. Additionally, no visual ablation studies are provided to support the design choices.

2. The identity used in the first figure is not appropriate, as it appears overly sexualized; more formal examples should be used.

3. The supplementary material is incomplete, lacking videos corresponding to all figures in the main text and appendix, which is insufficient.

4. The current evaluation metrics do not adequately assess body motion and audio alignment, despite this being a major claim of the paper.

### Questions
How is the pose acquired when only audio is provided?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors proposed a Audio-Pose Prior Refocusing mechanism that explicitly models prosody as a frame-level control signal to adaptively modulate pose embeddings and a Frame-Wise Audio–Video Interaction strategy in which an Audio DiT adapter refines audio using the current video context and the refocused pose prior.

### Strengths
The Frame-Wise Audio–Video Interaction strategy is interesting, where an Audio DiT adapter refines audio using the current video context and the refocused pose prior, producing pose and video-aware audio features that strengthen audio–motion coupling.

### Weaknesses
Major 
- The quantitative numbers are not too encouraging, the improvement is marginal in many metrics. At multiple places bold numbers are wrongly marked e.g. Table 4 and 5 FVD values.
- The visual comparison results in the supplementary are limited, under more demo cases folder what are the inputs?, is not clear. 
- Almost in all cases the person is in front of static background. What about those scenarios where e.g. person is giving a talk walking left right on the stage. 

Minor
- Consider citing relevant works like DisFlowEm : One-Shot Emotional Talking Head Generation using Disentangled Pose and Expression Flow-Guidance WACV 2025

### Questions
Please refer to the weakness section

### Soundness
3

### Presentation
3

### Contribution
3
