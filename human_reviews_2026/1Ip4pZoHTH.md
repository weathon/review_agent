# PACE: Part-Wise Slow-Fast Conditioning for Dance-to-Music Generation

- Decision: Reject
- Scores: 4, 4, 4, 6, 4

## Abstract
Dance-to-Music generation aims to compose music that is rhythmically aligned with human dance movements. While recent diffusion-based approaches have achieved promising results, they treat the dancer's body as a holistic unit when extracting motion features, thereby overlooking the fine-grained rhythmic contributions of individual body parts and the heterogeneous temporal dynamics manifested in both slow and fast motion patterns. In this work,  we approach the dance-to-music generation task from a fresh conditioning encoding viewpoint, where part-wise motion energy decomposition and a hierarchical slow-fast conditioning encoder are integrated to generate the conditioning for music latent diffusion. Through comprehensive subjective and objective evaluations of rhythm synchronization and generated music quality, experimental results on the AIST++ and TikTok benchmarks confirm that our framework consistently outperforms existing state-of-the-art approaches for dance-to-music generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes PACE, a conditioning encoding method for dance-to-music generation. It decomposes 3D pose data into part-wise slow and fast motion energy components, combines them with semantic features using a hierarchical encoder (HSF-Encoder), and utilizes this representation to condition a latent diffusion model. While this paper presents a novel perspective on dance-to-music generation, the empirical validation of the proposed method's effectiveness is insufficient.

### Strengths
1. The paper addresses the challenge of rhythm alignment from a novel standpoint, distinctly considering both global and local rhythmic structures.
2. The part-wise extraction of motion information is well-motivated, as it potentially captures fine-grained rhythmic details that are often overlooked when treating the human body as a single holistic entity.
3. The paper provides an intuitive visual analysis of the task and the generated results.

### Weaknesses
1. The paper lacks demonstration results (e.g., audio/video demos) for both the proposed method and the comparative methods, making it difficult to intuitively assess its effectiveness.
2. A comparative analysis against several recent state-of-the-art methods [1][2][3] is missing.
3. The work does not address the critical issue of genre matching between the input dance motion and the generated music.
4. A key claimed contribution is the extraction of slow and fast motion features. However, the paper fails to provide results that specifically evaluate the method's performance on slow-tempo and fast-tempo alignment independently.

### Questions
1. How exactly is the slow-fast conditioning representation $H$ injected into the generative network? (e.g., via cross-attention, concatenation, or another mechanism).
2. What are the frame rates (or temporal resolutions) of the input conditioning signal and the music features in the latent space?
3. Given the relatively small model size and training dataset reported, how does the proposed method address the issue of generative diversity in the output music? Have the authors considered evaluating diversity using established metrics, such as the Inception Score (IS)?
4. Regarding the rhythm evaluation metrics: What is the tolerance window (e.g., in milliseconds) used for beat alignment? A 1-second tolerance, as used in some early works, is arguably too permissive for a meaningful assessment of rhythm alignment. The paper seems to adopt this large tolerance, which warrants clarification and justification.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a method for music synthesis from dance movements (D2M). The authors raise a notable limitation in previous works regarding the body pose representation. They claim that treating a body pose as a whole-single holistic unit might cause to neglecting specific body parts that are more significant to infer the rhythm.
 
To address this, the authors propose to decompose a body pose representation into 1) slow and fast movements (referred as `kinetic energy’), and 2) part-level encoding, which treats each body part individually (five in total). The authors start by presenting a detailed example to enhance their motivation.
 
In the core of their work is PACE, a latent-diffusion model that produces musical audio, given a decomposed dance motion and visual features from a video. To ensure stability in the latent space, the authors separate the training process into VAE encoder-decoder training phase, followed by audio denoising training phase.
 
In addition, the authors propose a sophisticated encoding strategy the captures both overall pose semantics and part-level features.

### Strengths
The overall task is interesting and well-motivated, addressing a relevant and promising research direction.

The proposed idea of decomposed motion encoding is well-defined, insightful and intuitively appealing, with potential for improving motion understanding and synthesis.

The Methodology and Experiment sections are well-written, highly detailed and clear.

The authors conduct extensive comparison against previous works, including eight different evaluation metrics, and a user study to enhance the superiority of their method.

### Weaknesses
The use of I3D features and visual conditioning in general leads to two major limitations that are not addressed by the authors:
-	The fact that the model relies on video features forces it to depend on video input, making it incompatible with optical or inertial mocap data.

-	when the input is a video, the motion information is inherently prone to error caused by AlphaPose or MotionBERT, compared to that obtained from optical/inertial motion capture.

Qualitative results: The supplied plots are not convincing that PACE generates plausible music. No actual music examples are supplied (as far as I know).

The authors claim to propose a novel encoding technique but apply it only on their proposed model. An appropriate analysis would be to use it on other models/tasks as well.

The authors do not report any information about the genre diversity of PACE, or in-the-wild scenarios (e.g. out-of-distribution inputs).
The authors do not provide generated audio examples, which raises doubts about the effectiveness of the proposed method.
In general, the authors do not discuss limitations of their approach nor directions for future work.

Minor concerns:
The Introduction section is poorly written and contains irrelevant information. The figure is overly described and difficult to follow. Consider moving it to a later section where it is more appropriate.

L200: what is \mathcal{M}? not explicitly defined.

L200,215: \mathcal{D} is used for two definitions.

L307: It seems that the authors meant $n$ instead of $t$.

L376: missing space ``… (a)Objective’’.

For better readability, consider highlighting the best value for each metric in all tables.

### Questions
I would appreciate if the authors addressed the following concerns:

Qualitative results: The supplied plots are not convincing that PACE generates plausible music. No examples are supplied.

Ablation study: PACE-H and PACE-G are extremely important for the ablation study. However, the contribution of each is not quite clear from table 2 and is not discussed. Could the authors clarify the extreme drops and inconsistency (e.g. TIKTOK/BHS, AIST++/HSD)?
Objectives: The paper lacks formal definitions of losses for each training phase.

Could the authors elaborate on the perceptual and the patch-based adversarial losses (Section 3)?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a part-wise slow-fast conditioning encoder for dance-tomusic generation. The suggested method tries to explicitly decomposes part-wise motion signals into slow and fast dynamics, and introduces a hierarchical slow-fast conditioning encoder for diffusion-based music generation.

### Strengths
The motivation of the paper is clearly described, and the overall organization is good. 

The method is validated on two datasets, and the results outperform the comparative methods. 

The ablation study effectively validates the contribution of the proposed components.

### Weaknesses
What is the rationale for fusing the Fast and Slow components to serve as the condition for the Diffusion model? Given that the motion energy is decomposed into slow and fast components, why not use these decoupled features directly as the condition for the Diffusion model instead? This design choice requires further clarification.

How does the efficiency of the proposed method compare to other non-regressive models? The paper should address this, as the suggested method introduces several additional preprocessing steps, such as reconstructing 3D poses through MotionBert, partitioning them into five body parts, and computing them in two frequency bands using a Butterworth filter. Furthermore, it adds the Hierarchical SlowFast Conditioning Encoder. These additions likely incur a significant computational cost, and an efficiency comparison is necessary to properly evaluate the method’s practicality.

The paper would be strengthened by a more thorough theoretical analysis of the role and effect of decomposing motion energy into slow and fast components. While the experiments show it works, a theoretical explanation would provide deeper insight and a more solid foundation for the approach.

### Questions
The author needs to respond to the Weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes PACE (Part-wise Slow-Fast Conditioning Encoding), a diffusion-based framework for dance-to-music (D2M) generation. Unlike previous approaches that treat the dancer’s body as a holistic entity, PACE decomposes motion into body-part-specific slow and fast components using a Butterworth filter and encodes them hierarchically with a Hierarchical Slow-Fast Conditioning Encoder (HSF-Encoder). These conditioning features, together with visual features from an I3D encoder, guide a latent diffusion model trained in the audio latent space via a pre-trained VAE. Experiments on AIST++ and TikTok datasets show that PACE achieves better synchronization and perceptual quality compared to several methods.

### Strengths
1. The idea of part-wise slow-fast motion decomposition is both intuitive and technically sound, addressing the long-standing limitation of holistic motion modeling in D2M tasks.
2. The paper effectively justifies why different body parts and motion frequencies contribute distinct rhythmic information and demonstrates how this decomposition benefits generation.
3. Visualization of rhythm curves provides intuitive evidence of improved alignment between generated music and dance movements.

### Weaknesses
1. The concept of decomposing the body into parts is widely used in text-to-motion and music-to-dance, e.g. Bailando[1], AttT2M[2], ParCo[3], but the paper does not discuss related works, leaving the origin of the idea unclear.
2. The pipeline heavily relies on several external networks (AlphaPose, MotionBert, I3D, VAE). This raises questions about robustness, generalizability, and how errors from these modules propagate.
3. Although ablation studies are provided, it remains somewhat unclear how much each module (e.g., joint-level semantic encoding) individually contributes beyond aggregated effects.

\
[1] Siyao, Li, et al. "Bailando: 3d dance generation by actor-critic gpt with choreographic memory." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[2] Zhong, Chongyang, et al. "Attt2m: Text-driven human motion generation with multi-perspective attention mechanism." *Proceedings of the IEEE/CVF international conference on computer vision*. 2023.

[3] Zou, Qiran, et al. "Parco: Part-coordinating text-to-motion synthesis." *European Conference on Computer Vision*. Cham: Springer Nature Switzerland, 2024.

### Questions
1. The idea of decomposing the body into parts is widely used in text-to-motion and music-to-dance. What’s the difference between your part division method and  these existing part separation strategies? Could you compare the performance of your part division with these existing part separation strategies?
2. How were the hyperparameters of the Butterworth filter (e.g., cutoff frequencies for slow and fast components) selected? Did you perform any sensitivity analysis to assess how robust the model is to these choices?
3. Have you considered alternative groupings of body parts, such as upper body vs. lower body, instead of fully part-wise decomposition? How might this impact generation quality, rhythm alignment, or model complexity?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the problem of dance-to-music generation, where the input is a sequence of 3D or 2D dance motion and the output is the corresponding background music (audio).

The authors propose a diffusion-based pipeline. In particular, they decompose the motion representation into a slow band and a fast band, and combine them with semantic features as inputs to the diffusion model for audio generation.

The most novel part of the pipeline lies in this slow–fast motion decomposition.

According to the experimental results, the method achieves state-of-the-art performance on quantitative metrics, and the ablation studies are relatively thorough.

However, perhaps the main limitation of this work is the lack of subjective evaluation. For example, there are no video demo results, making it difficult to judge the perceptual quality of the generated music. According to this, we cannot provide a positive score this round. If the authors revise with some promising video demo I would consider improving the score.

### Strengths
**Technical Comments**

1. The decomposition of motion into “slow” and “fast” components is very interesting.

2. Using MotionBERT encodings as queries and keys, and the slow/fast encodings as values, is a clever and generally well-motivated design choice.

3. The subsequent cascade transformer architecture is also reasonable and fits well within the overall framework.

**Experiments**

1. The metric design is appropriate.

2. The comparative experiments are comprehensive.

3. The ablation study is basically sufficient, though there are several aspects that could be further supplemented.

### Weaknesses
**Demo Video**

I could not find any demo videos or links showing the generated results, which makes it difficult to assess the perceptual quality of the outputs. Therefore, **at the current stage, I am unable to assign a positive score**. However, **if the authors provide convincing demo results in the rebuttal showing high-quality generation, I would consider increasing my score**.

**Technical Comments**

1. Slow–Fast Decomposition.
I appreciate the idea of decomposing motion into slow and fast bands; however, the paper does not clearly explain what these bands subjectively represent in a given motion sequence. An example illustrating what constitutes the “slow” and “fast” components would help clarify this concept.
Moreover, although the authors provide an ablation study, it is still unclear what specific aspects of the generated music are influenced by the slow or fast motion features. As a result, the paper’s main novelty—this decomposition—feels insufficiently explored and presented.

2. Slow–Fast Fusion.
I am also curious about the learned trade-off parameter (α) used in the slow–fast fusion module, which seems to function as a soft mask. It would be informative to visualize or analyze this mask, showing its values and variations across different cases, and to provide an intuitive interpretation of its role in the system.
For instance, theoretically, if α = 1, the model relies purely on slow motion—what would that mean in practice? What is the average α value over the test set (e.g., AIST++)?

3. QKV Design.
Using MotionBERT encodings as queries and keys and the slow/fast motion encodings as values is an interesting design. However, it lacks intuitive explanation and supporting ablation studies. For example, how would the results change if slow, fast, and semantic (MotionBERT) features were fed as parallel inputs to the diffusion model? Why is this specific QKV configuration necessary?

**Writing**

Overall, the writing is clear and easy to follow. However, the background section feels somewhat lengthy, while some of the more critical technical components would benefit from deeper discussion and analysis.

### Questions
See Weakness.

In addition, whether the proposed system is capable of generating background music **with lyrics or vocals**, or if it is limited to purely instrumental audio. Clarifying this aspect would help define the scope and potential applications of the method.

### Soundness
3

### Presentation
2

### Contribution
2
