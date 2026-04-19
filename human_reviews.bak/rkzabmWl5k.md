# Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Recent advances in latent diffusion-based generative models for portrait image animation, such as Hallo, have achieved impressive results in short-duration video synthesis. In this paper, we present updates to Hallo, introducing several design enhancements to extend its capabilities.First, we extend the method to produce long-duration videos. To address substantial challenges such as appearance drift and temporal artifacts, we investigate augmentation strategies within the image space of conditional motion frames. Specifically, we introduce a patch-drop technique augmented with Gaussian noise to enhance visual consistency and temporal coherence over long duration.Second, we achieve 4K resolution portrait video generation. To accomplish this, we implement vector quantization of latent codes and apply temporal alignment techniques to maintain coherence across the temporal dimension. By integrating a high-quality decoder, we realize visual synthesis at 4K resolution.Third, we incorporate adjustable semantic textual labels for portrait expressions as conditional inputs. This extends beyond traditional audio cues to improve controllability and increase the diversity of the generated content. To the best of our knowledge, Hallo2, proposed in this paper, is the first method to achieve 4K resolution and generate hour-long, audio-driven portrait image animations enhanced with textual prompts. We have conducted extensive experiments to evaluate our method on publicly available datasets, including HDTF, CelebV, and our introduced ''Wild'' dataset. The experimental results demonstrate that our approach achieves state-of-the-art performance in long-duration portrait video animation, successfully generating rich and controllable content at 4K resolution for duration extending up to tens of minutes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work presents an enhanced version of the prior work Hallo. This work improves upon the previous model in terms of long-term video generation, 4K high-resolution video generation, and semantic textual label condition. This work is the first one to study 4K audio-driven portrait image animation. Experiment results show that this method achieves state-of-the-art performance in long-term portrait video animation, which can generate rich and controllable 4K content of tens of minutes.

### Strengths
1. This is the first work which attempts to animate portrait videos under 4K resolution. The qualitative results shown in the supplementary material demonstrate the improvements against the baseline.
2. The one hour length portrait video animation is interesting. The Gaussian noise augmentation technique is effective in alleviating the error accumulation issue in the autoregressive long video animation process.
3.  The textual prompt condition helps maintain the emotion and expression of the talking person, which is simple yet effective.

### Weaknesses
1. This work is a combination of existing works. The autoregressive animation pipeline has already been explored in EMO. The noise simulation trick is commonly used in the autoregressive generation models to alleviate error accumulation. The 4K resolution generation is an extension of VQGAN in the audio-driven portrait animation. Therefore, this work is a combination of prior works from the engineering perspective, which is not novel or significant enough.
2. Compared with Hallo, the animation framework remains unchanged and is quite similar to prior works like EMO. The reference unet borrows ideas from human image animation but the authors didn’t cite them properly. Please consider cite [1,2] in the final version.
3. The long video results shown in the supplementary do not contain too many translation and rotation of the talking head, which cannot demonstrate the temporal consistency for long-term animation. It is recommended to conduct such an experiment with diverse translation and rotation of the talking head to prove the effectiveness of this proposed method.
The ablation of HRE is not comprehensive enough. Comparisons with the 512x512 model cannot show the necessity of developing such a VQGAN-based superresolution in the pipeline. It is recommended to compare the proposed method with previous superresolution models, such as ESRGAN[3].

[1] Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation
[2] MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model
[3] Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

### Questions
No questions, please see the weaknesses for rebuttal.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the first method to generate an hour-long, audio-driven portrait image animation at 4K resolutions coupled with textual prompts. To accomplish this goal, the paper applies patch-drop augmentation on Gaussian noise, a vector quantization on latent codes with a high-quality decoder, and uses enhanced textual labels as the conditional inputs. These improvements constitute the proposed contributions.

### Strengths
The paper offers several contributions that are deemed to be the strengths of this paper. This work is an extension of a previous work on Hallo, thus termed Hallo2. It still attempts to propose a few enhancements that can further address the limitations of the previous work, mainly by extending it to long-hour portrait image animation with high resolutions. 

The patch-drop data augmentation looks interesting. It intentionally adds corruption to the appearance of the conditional frames while preserving the motion information. Meanwhile, the appearance contamination is resolved using the Gaussian noise as the other data augmentation. 

The 4K resolution was achieved by leveraging the VQ-GAN and code sequence prediction (existing but applied to a new task) to ensure smooth transitions among the temporal frames. 

Therefore, the main strengths of this paper would lie in the patch-drop augmentation, gaussian noise augmentation and the code sequence prediction.

### Weaknesses
The main idea of patch-drop data augmentation is to separate the motion from the appearance, where the former is controlled by temporal dynamics in the preceding frames. The patch-augmentation was applied to individual latent images z_{t-i}, where it was further partitioned into the K non-overlapping patches. Each patch was randomly masked. The question here is if we generate random masking and apply that to the time step t-1, would the same masking positions be enforced with the t-2 to t-N? Or the t-2 to t-N time steps still use a random masking operation. Could you clarify that and also show the motivation or the merits of each design? The other question is that it makes sense the omission of patches can still preserve the coarse structure. The motion dynamics on the other hand are not affected by the masking operations. Could you provide more details on why this is the case? 

Another intriguing part is related to the choice of patch size for dividing the latent image. Based on the results given in Table 3, the best patch size is 1*1, which means that a single pixel is dropped. This seems to contradict the intuition of using a patch. I don't mean 1 pixel is not a patch, but it does not conform to the normal choice of patch size. Could you offer more insights on why 1 pixel is considered to be more effective in this task? This would be very important. 

Regarding the preceding generated frames, which approach was used? Somehow, I am not able to find that detail in the paper, or maybe I miss it. Was that achieved by generating the second frame based on the first one, and then continually generating the third frame based on the first two frames? 

The design of Gaussian noise augmentation is simple and easy to understand, which was implemented by adding the Gaussian noise to the augmented latent representations. This was used as one of the conditional inputs. Could you provide more insights on why such an operation works? Ensuring temporal consistency is difficult. The paper seems to offer a much simpler and non-complex solution. Therefore, it would be more necessary to provide stronger validations on such a design. 

The set of conditional inputs used in this work includes four parts, which are the audio, textual, noise-augmented motion frames, and reference image encoding. Based on the flowchart given in Figure 3, the cross-attention conditions occur in different layers of the encoder and decoder of the U-Net. Could you justify how to select the corresponding layer for the designated condition? Are they all implemented via cross-attention? 

The high-resolution enhancement is still unclear to me. The paper does not seem to apply the super-resolution module but relies on the output from the diffusion model’s decoder and then a low-quality decoder. The technical part description in Sec 3.2 looks more like normal QKV operations. How the 4K resolution was achieved is still unclear, nor it was depicted in Figure 3. 

The other issue might be related to the over-claim of the proposed approach by saying the "hour-long". The evaluation was done on minutes duration (e.g., 4 minutes in evaluation). More evidences are required for this "hour-long" claim. 

The textual prompt implementation looks just like a regular conditional injection. This paper claims it to be one of the strengths of this work. 

The paper needs to add some more comparisons such as the EMO work. This has been mentioned in the Introduction but appears to be not compared in the experimental section.

### Questions
Please carefully address the above questions. I think there are many details that need to be clarified, particularly in the individual components constituting the proposed method.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Hallo2 for audio-driven face image animation, which mainly includes three training strategies. Firstly, this paper proposes a patch-drop data augmentation technique to prevent contamination of appearance information from preceding frames. Secondly, this paper extends the Vector Quantized Generative Adversarial Network discrete codebook space method for code sequence prediction tasks into the temporal dimension. Finally, this paper also introduces additional textual prompts for semantic control.

### Strengths
1. This paper discusses how to generate high-quality animated videos driven by audio, which is a very interesting topic.
2. The structure of this paper is well-organized and easy to follow. 
3. The experimental results show the effectiveness of the proposed method.

### Weaknesses
There are some questions and suggestions,
1. This paper proposes three strategies to improve the performance of audio-driven face image animation; however, the motivation for each strategy is not well explained. For example, in Section 3.2, a codebook prediction approach is used to enhance temporal coherence. It would be helpful if the authors provided more insights into why temporal coherence is insufficient without this codebook prediction approach.
2. This paper presents three distinct strategies; however, the connections between each contribution are weak, making it difficult for reviewers to grasp the central contribution. This lack of coherence also gives the impression that each contribution is only incremental.
3. From lines 238 to 241, the author uses a masking strategy to mask frames in the video. The author claims that this approach improves consistency. However, I’m a bit puzzled. In video generation, the masking strategy is generally beneficial for generating long videos autoregressively, so why would masking improve consistency? Could you please give more insights?
4. Some descriptions in the HIGH-RESOLUTION ENHANCEMENT section are a bit confusing. In the first half of this section, it seems that a Transformer is used to model the video in a continuous latent space. However, Figure 4 actually shows a codebook. Since it appears the authors are using a codebook, I suggest reorganizing the first half of this section to reflect this.
5. I observed flickering in some of the videos in the experimental results, especially with the man wearing a checkered shirt. Could this be due to limitations in the pre-trained VAE? Given that there is temporal alignment in the HIGH-RESOLUTION ENHANCEMENT section, why hasn’t this temporal alignment helped reduce the flickering effect?

### Questions
Please see above. If the author solves my problems, I will consider raising the score. Thanks.

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
4

### Summary
This paper presents a pipeline for generating long-duration, high-resolution, audio-driven facial animations. To enable extended video generation, the authors introduce patch-drop and Gaussian noise augmentation techniques during training. For high-resolution output, the method incorporates a VQ-based super-resolution stage and temporal alignment attention to ensure temporal consistency. Additionally, a norm-based textual prompt control mechanism is used to enhance the expressiveness of facial animations.

Experiments across multiple datasets demonstrate the effectiveness of the proposed approach.

### Strengths
- An effective augmentation strategy  which do not need to change model structure and support long-duration
    
- The textual condition injection propose a novel method to inject text prompt to a reference-net based animation structure.

### Weaknesses
- For high-resolution enhancement, the paper lacks a comparison with video VQ models (e.g., MagVITv2) and does not include quantitative evaluations of the super-resolution results from the SR encoder-decoder.
    
- For textual prompt control, the paper does not provide a detailed description of how prompts are used in comparisons with baseline methods or how prompts are incorporated during training data preparation. Additionally, details regarding the CFG process for managing multiple conditions (e.g., text, audio, reference image) are missing.

### Questions
- For patch augmentation, could the authors analyze the “effective augmentation ratio”? Since the purpose of patch augmentation is to encourage the model to disregard appearance information and focus on motion, an analysis of which patches contain motion information and which contain appearance information, along with a calculation of the effective augmentation ratio for the video clip, would provide valuable insight into the effectiveness of this design.
    
- How is the textual condition used during inference? Since it serves as an additional conditioning factor, the authors should clarify how this information is labeled, details of training behavior (e.g., random dropping for CFG), and how CFG is applied during inference. It would also be useful to include an analysis of the impact of conflicts between non-textual and text-audio conditions.
    
- On the HDTF dataset (i.e., Table 1), the proposed method outperforms most methods except Audio2Head. Could the authors provide an explanation for this? The paper should include further explanation of the evaluation metrics used to help readers understand which capabilities are being measured and improved.
    
- In the third image on the bottom right side of Figure 6, there appear to be “water droplet-like” artifacts. Could the authors explain the cause of these artifacts?

### Soundness
3

### Presentation
3

### Contribution
3
