# Personalized Facial Expressions and Head Poses for Speech-Driven 3D Facial Animation

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 3, 5, 5

## Abstract
Speech-driven 3D facial animation aims at generating facial movements that are synchronized with the driving speech, which has been widely explored recently. Existing works mostly neglect the person-specific talking style in generation, including facial expression and head pose styles. Several works intend to capture the personalities by fine-tuning modules. However, limited training data leads to the lack of vividness. In this work, we propose AdaMesh, a novel adaptive speech-driven facial animation approach, which learns the personalized talking style from a reference video of about 10 seconds and generates vivid facial expressions and head poses. Specifically, we propose mixture-of-low-rank adaptation (MoLoRA) to fine-tune the expression adapter, which efficiently captures the facial expression style. For the personalized pose style, we propose a pose adapter by building a discrete pose prior and retrieving the appropriate style embedding with a semantic-aware pose style matrix without fine-tuning. Extensive experimental results show that our approach outperforms state-of-the-art methods, preserves the talking style in the reference video, and generates vivid facial animation. The supplementary video and code will be available at https://adamesh.github.io.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a novel architecture for speech-driven 3D facial animation amenable to pose and expression inputs. A series of encoders are properly combined to ensure the generation abides to the desired pose and facial expressions. The paper includes some novel components in the domain of face synthesis, namely the use of LoRA for convolutions to facilitate the training, and the use of a codebook with a PoseGPT to generate a plausible sequence of poses. The authors include a very strong supplementary material showcasing the benefits of the proposed approach, and a user-study to support the generated results.

### Strengths
The paper is overall well written and presented, and it is accompanied by a thorough supplementary material with a detailed video showcasing the results. The amount of work put towards completing the paper is significant and valuable. 

- The attempt to encode facial expressions and pose while synthesizing 3D faces from speech is novel as far as I am aware, and the results are generally remarkable.

- The use of a VQ-VAE to quantize poses and a generative method to generate these from the initial latent vector is also novel and (seems to be) technically sound. This generative approach from a codebook is also well-grounded and allows for generating meaningful poses across sequences. 

- The use of LoRA to facilitate the learning is also novel in this context, and while I am not sure such plug and play can be counted as a contribution, its adaptation to convolutions is interesting.

### Weaknesses
In my opinion, there are only few aspects that could need further consideration:

1.) It is my understanding that the facial expressions are passed as input to the network, is that correct? If this is the case, I would like the authors to provide some visual examples of the same inputs where the target facial expressions are different. I wonder how this combines to the fact that speech is also a cue conveying emotions and thus cannot be 100% detached from facial expressions. If the input video is of a person in a state of sadness, how would the method perform when the reference expression is set to happy? How realistic would this be? If the expression is not directly inferred from the audio, then I think Fig. 1 needs a better representation and this to be properly referred to in the text.

2.) The method contains an identity encoder. I wonder how is this used when generating the outputs considering that the final result is a 3D mesh without texture. In the case that this affects the 3D mesh only, I think that a study showing whether humans can distinguish two different generated meshes of the same person from a third one is necessary. Generating 3D meshes without regards to proper identity preservation can lead to relatively poor user experience.

### Questions
I have included all my questions above in the weaknesses section. Please do address these.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes novel adaption strategy for capturing the reference emotion and pose styles in 3D talking head generation. Given a reference video and a driving speech, the proposed AdaMesh method can generate a 3D talking head saying the speech with dynamic emotions and poses similar to the reference video. The expression and pose are generated by separate network branches, which features mixture-of-LoRA (MoLoRA) adaption and discrete pose space, respectively.

### Strengths
### 1. Novel MoLoRA for personalized facial expression

LoRA may currently be the most appropriate technic for adapt a generalized model into personalized (or case-specific) ones. The verification of LoRA in the talking head field should benefit the community.

### 2.  Investigation of semantic-aware pose style

Pose functions as an information conveying modal in human communication, while the relation of speech semantic and head pose lacks investigation in the talking head/face researches. This paper propose a novel strategy to align semantic features into pose codes, in which the pose codebook can be established using large pose datasets without speech.

### 3. Better demo results than other compared works

The qualitative results are remarkably better than others in the demo video, including more realistic poses and prominent expressions.

### Weaknesses
1. I argue with the authors' claim that "**the first work that takes both facial expressions and head poses into account in personalized speech-driven 3D facial animation.**" Facial[r1] meets all the key words, **facial expression**, **head pose**, **personalized**, and **3D**. Besides, SadTalker[r2] implements specific branches for pose and expression generation for 3DMMs, which has not been cited. The SadTalker finally render 2D RGB videos from its generated 3DMMs, which I think cannot cover up that it's a 3D method and exclude it from the discussion in the reviewing paper. 
2. Although the generated poses are better than compared works, they are **not realistic enough to convey semantics**, thus I cannot distinguish whether the poses are semantic-aware as claimed in paper. The generated poses drive the shoulder with the same displacement and rotation with the head, while human rotate their heads with relatively fixed shoulders. The dynamics of neck needs more investigation. 
3. The pose adaption(retrieval) method means it **can only generate seen pose styles** in the training set.
4. The **sampling method** of Pose-GPT is missing.
5. Minor issue: in the last line of Sec. 4 Dataset, "We **introduce** VoxCeleb2 test dataset ..." I think "introduce" is inappropriate since the VoxCeleb2 is already mentioned for pose adapter pre-train.

[r1] Zhang, Chenxu, et al. "Facial: Synthesizing dynamic talking face with implicit attribute learning." Proceedings of the IEEE/CVF international conference on computer vision. 2021.   
[r2] Zhang, Wenxuan, et al. "SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
Do you assign a number to each semantic-aware pose style matrix S and transform this very number into a one-hot vector as pose style embeddings?

### Soundness
3 good

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This research introduces AdaMesh, a approach to speech-driven 3D facial animation that learns personalized talking styles from short reference videos, resulting inexpressive facial expressions and head poses. The proposed method, which includes MoLoRA for facial expression adaptation and a pose adapter with a semantic-aware pose style matrix, outperforms existing techniques.

### Strengths
1.AdaMesh addresses the limitation of existing works by focusing on capturing and adapting the individual's talking style, including facial expressions and head pose styles.

2.AdaMesh introduces a technique called mixture-of-low-rank adaptation (MoLoRA) for fine-tuning the expression adapter.

3.A pose GPT and VQ-VAE are used for the pose adapter.

### Weaknesses
1.The major contributions are MoLoRA for facial expression adaptation and pose GPT for pose adaptation. The two key points have no much correlation between each other, the combination of these two points makes this paper not well-focused.


2.Given the head pose training data from VoxCeleb2, how do you extract the head pose information? The GT head pose in the demo video seems unstable and lacks time consistency. In such a condition, does the model learn to generate meaningful head pose? 
 
3.Does the authors try to train baselines, such as FaceFormer or CodeTalker, on their data?

4.Experiments on the classical datasets, such as the VOCA dataset, could further verify the model’s effectiveness. The results on one dataset can not prove the model’s generalization.

5.User study lacks credibility, more details about the user study setting should be clarified.

### Questions
see above

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper focuses on generating the talking face with a person-specific style, i.e., facial expressions and head poses. In the proposed method AdaMesh, the facial expression style is adapted by a mixture-of-low-rank adaptation. The pose style is adapted by matching the poses to discrete pose priors. Extensive experiments show that the proposed method preserves the talkers’ talking styles.

### Strengths
1. The proposed method preserves personalized talking styles, including facial expressions and head poses. 
2. Qualitative and quantitative results show the advantages of the proposed AdaMesh.

### Weaknesses
1. This paper uses LORA with multiple ranks to capture the multi-scaled features of facial expressions. It is unclear why the multi-rank structure could build the multi-scale representation. Please explain their correlations. 

2. This paper uses discrete poses as priors, and then during the inference this paper matches each pose into its nearest prior pose. It is unclear how to determine the number of prior poses. Intuitively, if the prior poses are sparse, it might influence the accuracy and smoothness of the generated poses. If the prior poses are dense, it requires plenty of training data to learn the generator for each pose prior. 

3. The introduction of PoseGPT is not clear. What is the task when PoseGPT is trained? Fig 2 shows that poseGPT is trained by predicting the poses (pitch, yaw, roll) according to the driving speech and pose style embedding. Are the input pose style embeddings with the same length as that of the predicted head pose sequences? If not, how should we determine the length of the predicted head pose sequence?

4. Some minors:
  - The first sentence in the paragraph about Pose GPT lacks a  comma or period.

### Questions
Please refer to the weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
