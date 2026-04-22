# AUHead: Realistic Emotional Talking Head Generation via Action Units Control

- Avg Score: 4.33
- Decision: Accept (Poster)
- Scores: 8, 4, 0, 6, 2, 6

## Abstract
Realistic talking-head video generation is critical for virtual avatars, film production, and interactive systems. Current methods struggle with nuanced emotional expressions due to the lack of fine-grained emotion control. To address this issue, we introduce a novel two-stage method (AUHead) to disentangle fine-grained emotion control, i.e. , Action Units (AUs), from audio and achieve controllable generation. In the first stage, we explore the AU generation abilities of large audio-language models (ALMs), by spatial-temporal AU tokenization and an "emotion-then-AU" chain-of-thought mechanism. It aims to disentangle AUs from raw speech, effectively capturing subtle emotional cues. In the second stage, we propose an AU-driven controllable diffusion model that synthesizes realistic talking-head videos conditioned on AU sequences. Specifically, we first map the AU sequences into the structured 2D facial representation to enhance spatial fidelity, and then model the AU-vision interaction within cross-attention modules. To achieve flexible AU-quality trade-off control, we introduce an AU disentanglement guidance strategy during inference, further refining the emotional expressiveness and identity consistency of the generated videos. Results on benchmark datasets demonstrate that our approach achieves competitive performance in emotional realism, accurate lip synchronization, and visual coherence, significantly surpassing existing techniques.
Our implementation is available at https://github.com/laura990501/AUHead_ICLR

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents AUHead, a two-stage framework for audio-driven emotional talking head generation. The first stage leverages a fine-tuned Audio-Language Model (Qwen-Audio-Chat) to disentangle facial Action Units (AUs) from audio using a spatial–temporal tokenization and Chain-of-Thought (CoT) based “emotion-then-AU” reasoning process. The second stage introduces an AU-driven diffusion generation framework, including AU-to-2D mapping, context-aware temporal embedding to synthesize identity-consistent and emotionally expressive faces. Experiments on MEAD and CREMA datasets show strong quantitative and qualitative performance improvements over state-of-the-art baselines (e.g., MEMO, HalloV1/V2), especially in emotional realism and AU alignment. Overall, the paper is technically novel, methodologically well motivated, and experimentally solid, offering an interpretable bridge between audio and visual modalities.

### Strengths
1.Introducing AUs as an intermediate control effectively bridges speech understanding and video generation.
2.Stage 1 pioneers the use of an ALM for AU generation, which is a highly insightful idea for a generalizable control signal. The “emotion-then-AU” CoT design and spatial temporal AU tokenization are elegant, showing how ALMs can perform emotion reasoning beyond conventional recognition tasks.
3.Stage 2's AU-driven controllable generation framework enhances emotional expressiveness while maintaining identity and lip sync.
4.The paper is clear, includes all implementation details, and provides an anonymous open-source link, which supports reproducibility.

### Weaknesses
1．The user study involved only 4 participants who are identified as "colleagues from our laboratory." This sample size is too small, and the lack of participant independence introduces a severe bias, which significantly diminishes the persuasiveness of the user preference results for a top-tier conference.
2．While the intuition behind using AUs as intermediate control is strong, the paper could better clarify why ALM-based AU prediction is preferred over simpler audio–AU regression networks.

### Questions
1.You are encouraged to expand the user study with a larger and more diverse set of participants to further validate the perceptual quality claims.
2.Can the AU guidance be manually adjusted at inference time to control or edit facial expressions—for example, by modifying AU values to vary the intensity or type of emotion?

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
5

### Summary
The paper proposes the AUHead method for the talking head generation task with fine-grained emotion control. This method is divided into two stages: in the first stage, an audio-language model (ALM) is used to predict Action Units sequences from audio; in the second stage, AU and Audio are fed into the diffusion model as conditions for video generation.​

### Strengths
- The paper proposes a method of using ALM for fine-grained emotion prediction, which is relatively rare in related works.​
- The paper conducts rich and comprehensive experiments.​
- The paper is well written and presents content clearly.​

### Weaknesses
- Although using ALM for emotion prediction is novel, the necessity of using ALM is questionable. The AU sequences used in the paper are only 24-dimensional, which is obviously a low-dimensional vector. There are many works on 3D talking head generation based on audio-to-mesh/3DMM that have achieved good results (many of which specifically model emotion), and such scenarios are similar to the problem mentioned in this paper, but even more difficult. The authors need to use experiments to demonstrate that the traditional audio embedding + regression method cannot predict AU sequences ideally. In addition, fine-tuning ALM has significantly higher inference cost and training difficulty, which limits this method to a certain extent.​
- The paper claims that the model has "speech understanding" ability, but it is hard to be convinced that AU sequence prediction is equivalent to "speech understanding".​
- The "AU-based disentanglement guidance strategy" proposed in the paper is not original. In fact, many multi-conditional generation tasks adopt the same or similar CFG variants, such as MagicInfinite[1] (eq 7.).

[1] Yi, Hongwei, et al. "Magicinfinite: Generating infinite talking videos with your words and voice." arXiv preprint arXiv:2503.05978 (2025).

### Questions
Please refer to the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes AUHead, a two-stage talking head generation method that leverages ALM and  facial action units to control the speaker's emotion.

### Strengths
1. The talking head generation task is of practical importance. Emotional control is a important topic in talking head generation since natural talking heads exhibit emotions.
2. The paper is easy to follow.

### Weaknesses
1. The performance is far from satisfactory. The results shown in demo are blurry. The expressions are unnatural.  All audio and source images shown in demo are from lab dataset MEAD[1] and CREMA[2], which cannot indicate that the model can generalize to more diverse inputs. In short, the performance is far from SOTA and is unacceptable for publication.
2. The paper lacks comparisons with recent emotional talking head methods, including EDTalk[3], EAT[4]. Lack of comparisons makes it difficult to validate the effectiveness of the proposed method.
3. Utilizing ALM and AU to control emotion is of little significance now. Recent methods, like omnihuman1.5[5], can control emotion with diverse text. Diverse text achieves more powerful control than AUs.

[1] Wang, Kaisiyuan, et al. "Mead: A large-scale audio-visual dataset for emotional talking-face generation." European conference on computer vision. Cham: Springer International Publishing, 2020.

[2]  Cao, Houwei, et al. "Crema-d: Crowd-sourced emotional multimodal actors dataset." IEEE transactions on affective computing 5.4 (2014): 377-390.

[3] Tan, Shuai, et al. "Edtalk: Efficient disentanglement for emotional talking head synthesis." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[4] Gan, Yuan, et al. "Efficient emotional adaptation for audio-driven talking-head generation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[5] Jiang, Jianwen, et al. "Omnihuman-1.5: Instilling an active mind in avatars via cognitive simulation." arXiv preprint arXiv:2508.19209 (2025).

### Questions
Can the method generalize to audio and source images that are not from MEAD and CREMA?
Can the authors provide comparisons with more recent methods?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces AUHead, a novel framework that creates emotionally expressive, lip-synced talking-head videos by extracting facial Action Units from audio. It employs an Audio-Language Model to derive AUs and a diffusion model to animate faces, surpassing previous methods in emotional realism, lip sync accuracy, and visual coherence.

### Strengths
- The idea of using AUs as a bridge for generating expressive talking-head videos is novel and could lead to more controllable and interpretable results in audio-driven face generation.
- The disentanglement of AUs from audio before the generation process is an interesting and effective way to tackle the complexities of fine-grained facial expression modeling.

### Weaknesses
- While the model performs well in terms of lip sync and expression fidelity, the paper doesn't sufficiently address how well the model handles longer videos or more complex facial movements. Temporal consistency issues may arise in longer video synthesis.
- The spatial-temporal tokenization approach to generate AU sequences may introduce issues related to the loss of subtle details in facial expressions, especially when reducing the sequence length to manageable tokens (as discussed in the sparse tokenization section).
- While AUs provide a more interpretable control space compared to emotion labels, the complex tokenization and disentanglement from audio may reduce the overall interpretability of how specific AUs contribute to particular facial expressions. A clearer explanation of the model’s inner workings could help with understanding the generated results better.
- Although the paper compares AUHead to state-of-the-art methods, further comparisons with models that also leverage other types of intermediate representations (e.g., motion priors, landmarks) could give a better perspective on AUHead’s relative strength.

### Questions
- Does the model perform equally well on unseen audio samples not part of the training data, especially in terms of subtle emotional cues that were not explicitly included in the training set?
- How does the model handle cases where there are multiple potential AU sequences for a given audio input? Is there any inherent ambiguity in predicting AUs, especially in more complex emotional expressions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents AUHead, a two-stage framework for audio-driven talking head generation that aims to improve the expressiveness and controllability of facial animations. Instead of directly generating video from audio and a reference image, the proposed method first extracts Action Unit (AU) sequences from the input speech using an audio language model (ALM), leveraging its capability to capture emotional and articulatory cues. In the second stage, a pretrained diffusion model is enhanced with an AU adapter module that integrates AU embeddings, audio features, and the reference image through cross-attention mechanisms. Experiments on benchmark datasets (MEAD and CREMA) show that AUHead generates videos with improved facial expressiveness and temporal smoothness compared to existing methods.

### Strengths
* Enhanced Facial Expressiveness: By using Action Units (AUs) extracted from audio as an intermediate representation, AUHead generates facial animations with richer and more realistic emotional expressions. AUs provide structured, interpretable cues for specific facial muscle movements, enabling the model to produce more nuanced and natural-looking expressions compared to baseline methods.
* Effective Utilization of Pre-trained Models: The method leverages a audio language model to predict AU sequences from speech. This design capitalizes on the strong audio understanding capabilities of existing pre-trained models, making the AU prediction stage more effective and avoiding the need to learn complex audio-to-AU mappings from scratch.

### Weaknesses
* The experimental setup and comparisons are not sufficiently rigorous:
    * There has been extensive research on facial expression control in the audio-driven talking-head generation field, including but not limited to [1, 2, 3, 4, 5, 6]. However, this work does not compare with any of these methods in its experiments. What is the reason for this omission?
    * The datasets used in this work are MEAD and CREMA, which contain only 60 and 91 identities respectively. The limited scale and quality of the experimental data significantly weaken the validity and persuasiveness of the evaluation.
    * The comparison lacks up-to-date state-of-the-art methods. The baselines chosen are at least half a year to a year old; for instance, EchoMimic[7] has already released version V3 and the Hallo[8] series has reached V4. This work fails to discuss or compare against more recent approaches such as Sonic[9], MultiTalk[10], and WanS2V[11].
* Insufficient experimental results: The quantitative results of this work leave room for improvement. Notably, in terms of lip-sync accuracy, it ranks only sixth on the MEAD test set—lower than the pre-trained method MEMO[12]—suggesting that the proposed AUHead may degrade the performance of the underlying pre-trained model. Moreover, the provided visual results exhibit clear issues, such as inaccurate lip movements and blurred teeth, indicating a noticeable gap compared to current leading audio-driven methods like MultiTalk and WanS2V.
* Practical challenges arising from the core methodology: Recent advances in large video generation models have enabled effective text-guided facial expression control via implicit semantic understanding—for example, methods like InfiniteTalk and FantasyTalking2 leverage vision-language models (VLMs) to generate emotion labels and achieve expressive control without explicit supervision. In contrast, this work relies on constructing explicit Action Unit (AU) labels for emotion control, which requires large-scale supervised training data annotated with AUs. This approach faces significant practical challenges in data collection and annotation, making it less scalable and deviating from mainstream trends. As a result, its potential contribution to the broader research community appears limited.

[1]Gan, Yuan, et al. "Efficient emotional adaptation for audio-driven talking-head generation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2]Tan, Shuai, et al. "Edtalk: Efficient disentanglement for emotional talking head synthesis." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

[3]Ma, Yifeng, et al. "DreamTalk: When Emotional Talking Head Generation Meets Diffusion Probabilistic Models." arXiv preprint arXiv:2312.09767 (2023).

[4]Tan, Weipeng, et al. "Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation." arXiv preprint arXiv:2504.18087 (2025).

[5]Ma, Xingpei, et al. "Playmate: Flexible Control of Portrait Animation via 3D-Implicit Space Guided Diffusion." Forty-second International Conference on Machine Learning.

[6]Lin, Bin, et al. "Takin-ADA: Emotion Controllable Audio-Driven Animation with Canonical and Landmark Loss Optimization." arXiv preprint arXiv:2410.14283 (2024).

[7]Chen, Zhiyuan, et al. "Echomimic: Lifelike audio-driven portrait animations through editable landmark conditions." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 3. 2025.

[8]Xu, Mingwang, et al. "Hallo: Hierarchical audio-driven visual synthesis for portrait image animation." arXiv preprint arXiv:2406.08801 (2024).

[9]Ji, Xiaozhong, et al. "Sonic: Shifting focus to global audio perception in portrait animation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[10]Kong, Zhe, et al. "Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation." arXiv preprint arXiv:2505.22647 (2025).

[11]Gao, Xin, et al. "Wan-s2v: Audio-driven cinematic video generation." arXiv preprint arXiv:2508.18621 (2025).

[12]Zheng, Longtao, et al. "Memo: Memory-guided diffusion for expressive talking video generation." arXiv preprint arXiv:2412.04448 (2024).

### Questions
* Does AUHead support generating videos from the same audio segment guided by different emotion labels? If not, will inaccurate emotion recognition by the ALM lead to incorrect emotional expressions in the generated results?
* Can the core method of AUHead be equally applied to more recent frameworks such as Sonic[1] and WanS2V[2]?
* Could the experimental evaluation be expanded to more comprehensively demonstrate the effectiveness of the proposed method?
* The videos provided in the supplementary material are all shorter than 10 seconds. How does AUHead perform on audio clips longer than 10 seconds? Could the authors provide corresponding results?

[1]Ji, Xiaozhong, et al. "Sonic: Shifting focus to global audio perception in portrait animation." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[2]Gao, Xin, et al. "Wan-s2v: Audio-driven cinematic video generation." arXiv preprint arXiv:2508.18621 (2025).

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents AUHead, a two-stage framework for generating realistic and emotionally expressive talking head videos conditioned on speech and a reference image. The key idea is to employ Facial Action Units as an interpretable intermediate representation bridging audio and visual modalities, addressing the difficulty of producing fine-grained emotional expressions in prior works.
In Stage 1 (Understanding), an Audio Language Model (Audio-Qwen-Chat) extracts emotion aware AU sequences from speech through two steps: (1) a spatio-temporal AU tokenization strategy that compresses dense AU vectors, and (2) an emotion-then-AU Chain-of-Thought process that first predicts emotion categories before generating detailed AU sequences.
In Stage 2 (Controllable Generation), an AU-guided diffusion model synthesizes videos via three modules: (1) AU Representation, which upsamples and projects AU sequences into structured 2D facial maps (LMK or RoM) to ensure spatial realism; (2) Context-Aware AU Embedding, using temporal convolution to maintain motion consistency; and (3) AU Vision Interaction, which introduces AU-conditioned cross-attention and a classifier-free AU guidance mechanism for flexible control between expression fidelity and visual quality.

### Strengths
1. The supplementary visualizations are exceptionally clear and persuasive, providing strong qualitative evidence of the method’s superiority.

2. The two-stage pipeline is well justified, and adopting AUs as the intermediate feature is a sensible and effective choice.

3. As I know, this is the first work to leverage a large audio-language model to predict AU sequences from speech, enabling disentangled emotion-aware representations for talking-head generation.

4. Stage 2 effectively maps AUs to 2D spatial priors (landmark or mesh) and integrates temporal AU embeddings with AU–vision cross-attention adapters. 

5. This paper is well-orgnized and well written.

### Weaknesses
1. The AU regression in Stage 1 achieves an MAE of around 0.2, which is comparable to inter-annotator variability. However, the paper does not explicitly discuss how this residual AU prediction error might influence the final video generation quality. This limitation is acceptable, but a short analysis would further clarify the robustness of the pipeline.

2. The AU sequences are downsampled to 5 fps to fit within the ALM’s context window, which may constrain the modeling of subtle, high-frequency facial movements.

3. The user study is small in scale, this is not sufficiently rigorous for strong perceptual claims and risks bias.

4. Some implementation details are briefly described and could be clarified. It would be useful to explain how the AU sequences are converted into 2D facial representations (landmarks and RoM).

### Questions
1. Does the chosen set of 24 AUs cover all the major emotional expressions present in the MEAD and CREMA datasets?

2. In the spatial-temporal AU tokenization, the AU sequences are downsampled to 5 fps and encoded as (index, intensity) pairs. Could the authors explain how this 5 fps rate was determined and whether alternative rates were considered?

3. Please provide more technical details on the AU to 2D representation mapping (LMK and RoM).
 
4. The user study involves a small number of raters. Could the authors elaborate on the selection process and whether they plan to expand this evaluation in future work?

### Soundness
3

### Presentation
3

### Contribution
3
