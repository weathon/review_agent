# UniMMVSR: A Unified Multi-Modal Framework for Cascaded Video Super-Resolution

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Cascaded generation framework has emerged as a promising technique for decoupling the computational burden associated with generating high-resolution videos using large foundation models. Existing studies, however, are largely confined to text-to-video tasks and fail to leverage additional generative conditions beyond text, which are crucial for ensuring fidelity in multi-modal video generation. We address this limitation by presenting UniMMVSR, the first unified generative video super-resolution framework to incorporate hybrid-modal conditions, including text, images, and videos. We conduct a comprehensive exploration of condition injection strategies, training schemes, and data mixture techniques within a latent video diffusion model. A key challenge was designing distinct data construction and condition utilization methods to enable the model to precisely utilize all condition types, given their varied correlations with the target video. Our experiments demonstrate that UniMMVSR significantly outperforms existing methods, producing videos with superior detail and a higher degree of conformity to multi-modal conditions. We also validate the feasibility of combining UniMMVSR with a base model to achieve multi-modal guided generation of 4K videos—a feat previously unattainable with existing techniques.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces UniMMVSR, a unified multi-modal generative video super-resolution framework designed to upscale AI-generated low-resolution videos to ultra-high-resolution (e.g., 4K), supporting multi-model input as guidance, including text, multi-ID images, and reference videos.
The proposed solution includes the following parts:
- A unified conditioning strategy combining channel concatenation (for low-res video) and token concatenation (for reference modalities), with separated conditional RoPE for positional embedding.
- A SDEdit-based degradation pipeline that simulates realistic generative degradations of base models, combining with the conventional synthetic degradation baseline.
- A difficult-to-easy curriculum training strategy to align sub-tasks of varying complexity.

### Strengths
- This paper introduces multi-modal guidance into diffusion-based video super-resolution (VSR). The qualitative results demonstrate the effectiveness of the guidance in guiding the content generation with appealing performance.
- The ablation study shows the effectiveness of the proposed sdedit degradation as well as the the conditioning strategy for multi-modal guidance VSR

### Weaknesses
- The contributions of the paper seem to weigh more on the engineering side rather than the technical novelty. This paper seems to be built on lots of existing successful practices, like DiT, RoPE, cascaded modeling, etc. The improvements upon the above parts, e.g., conditioning strategy, degradation pipeline and training order seem kind of trival, though maybe effective.
- The fidelity of the model seems poor compared with previous baselines like SeedVR and STAR, which is obvious in nearly all the qualitative results. It is unclear if this is caused by the introduced multi-model guidance.

### Questions
My main concerns of this paper are as follows:

1. The two major weaknesses above, i.e., technical contributions and fidelity.

2. Some of the paper details are vague: 1) Achieving 4K SR is very VRAM expensive. The details of how to achieve such high-resolution video need further explanation, e.g., how many frames can be processed in one inference? how to process long videos? How much VRAM is required for each inference, etc? 2) The details of image-related transformations for reference alignment are vague in Line 299-300. What detailed transformations are used for training?

3. The complexity comparison is missing. There should be FLOPs, parameters and inference time comparison. The training time should also be provided.

4. The details of the evaluation data seem to be missing. What data is used for quantitative and qualitative comparison? The author may also consider making comparisons on commonly used synthetic and real-world benchmarks, following previous works such as Upsale-A-Video, STAR, and SeedVR, with standard metrics such as PSNR, SSIM, CLIP-IQA, MUSIQ, etc. This ensures a more straightforward comparison with previous methods.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a cascaded video super-resolution framework that can support unified inputs of text, images, and videos. To unify different conditions together, the paper explores options such as channel concatenation of low-resolution video, token concatenation of visual references, and separated conditional RoPE for multi-ID images and reference videos. The paper also designs a degradation framework to simulate the artifacts and distortion generated by the base model. The contributions of this work are the following. The proposed framework is the first unified one that can handle multimodal conditions. The paper also provides some useful insights on how to effectively utilize diverse inputs—including low-resolution video, text, multiple ID images, and reference videos. The experimental demonstrations also look solid as it provides sufficient visualizations.

### Strengths
The strengths of this work are outlined below. 

The proposed framework was claimed to be the first one that can handle multimodal conditions simultaneously for the video super-resolution task. 

The paper provides some useful insights on how to effectively utilize diverse inputs—including low-resolution video, text, multiple ID images, and reference videos. Considering the difference in modalities, the paper explores options such as channel/token concatenation and separated RoPE. 

The experimental demonstrations also look solid as they provide sufficient visualizations, where the ablation shows effectiveness for different components.

### Weaknesses
Though the paper offers the above strengths, it still demonstrates a few critical weaknesses, which need to be further clarified. 

1. The entire framework looks a bit complicated. For different modalities, it chooses different options such as channel/token concatenation and separated RoPE.  Can we also process them all as tokens and then use token concatenation? After all, the paper shows that the token concatenation performs the best on multi-modal condition injection when incorporating multiple ID images and reference videos. In this case, the motivation work regarding this part could be further clarified. 

2. The framework also relies on the choice of different training orders, basically on how to reorganize the order for different tasks of text-to-video, multi-ID image-guided, and video editing. This also depends on how to set up the probability for choosing different tasks. This makes the proposed framework not generalizable and could be difficult to train. It is unclear whether the distribution of these probability values will vary again if datasets and or some base architectures change. 

3. The paper should provide some computational complexity analysis on the proposed framework, as it consists of many different components. The most consuming part would lie in the training phase. But it will still be good to list the inference time and the used parameters, and so on. 

4. The reason for using multi-ID images is not well clarified. It is unknown which synthesis output relies on these multi-ID images. And among the provided multi-ID images, which image is placed with more emphasis? 

5. The proposed method does not seem to achieve the best among the compared methods. For some listed methods such as VEnhancer, STAR and SeedVR, they can all handle the tasks of Text-to-video Generation, Text-guided Video Editing, and Multi-ID Image-guided Text-to-video Generation. What are the advantages of the proposed framework compared to these methods from this perspective?

### Questions
1. When describing the technical details of low-resolution video via channel concatenation, it appears that "Upsampler" was directly applied to the latent code, other than in the pixel space. There could be some confusion here. 

2. The role of using multi-ID images needs to be better clarified. Based on the illustrations from Figure 3, why does the super-resolution on the text image rely on the input face image, e.g., the third row case? It appears that super-resolution output has nothing to do with the face image.

3. What is the computational complexity of the proposed method, and how reproducible is the proposed approach?

4. What is the motivation for not using the all tokens process for text, image, and videos? Has this been explored in this paper?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes a new model, UniMMVSR, that upscales AIGC videos while also respects hybrid conditions including low-res input, text, ID images and other videos. With multi-modal inputs, it is able to upscale a video up to 4K. It also shows a degradation pipeline based on SDEdit. It shows outstanding performance among multiple video generation tasks.

### Strengths
### Motivation
- UniMMVSR is designed for AIGC video super-resolution with a cascaded model after the base model. It shares the same latent space with the low-res videos.
- To obtain high-fidelity results, UniMMVSR uses multiple inputs as conditions. It unifies multi-modalities including texts, images and videos.  

### Method
- Token-wise concatenation makes sense for reference input, given that they are not pixel-aligned with the low-res input video.
- Separated conditional RoPE also makes sense for reference tokens. 
- To simulate AIGC artifacts, authors use SDEdit pipeline to "degrade" the input videos. 
- Reference augmentation is used to better adopt references in different scenarios. 

### Experimental results
- Qualitative results show UniMMVSR can preserve details from reference inputs in Fig. 3.
- UniMMVSR shows competitive qualitative results in three tasks on different no-reference metrics, especially in Multi-ID reference video generation. 

### Writing
The paper is easy to follow and well-written.

### Weaknesses
### Motivation
- The title sounds a little misunderstanding to me. UniMMVSR is designed to super-resolve AIGC videos, but authors use the term "Cascaded Video Super-Resolution", which is more often used for multiple, cascaded networks for video super-resolution in my opinion. Maybe use "Reference-based Video Super-Resolution" or other terms would be better than "Cascaded Video Super-Resolution".

### Method
- Since most of the VAEs are lossy, will "decoding LR latent -> pixel upscaling -> VAE encode" further increase the information loss? (Sec 3.2, Low-resolution video via channel concatenation).
- The token-wise concatenation is not well documented. What strategies are used if the number of reference tokens are not the same, i.e. variable token length? Are there "NULL" tokens for padding during the training/inference? 

### Experimental results
- 4K results have some limitations. (a) Too short, only 1s (or 21 frames). (b) Color shift (e.g., 41397180.mp4 in text-to-video generation. The first frame and the last frame look so different).
- The qualitative results are mainly compared on no-reference metrics such as MUSIQ, DOVER, etc, which are infamous for their bias towards their training data. Can authors show a user study as a supplement? 

### Writing
- Typos: "our UniMMVSR model also need to" -> "our UniMMVSR model also needs to".

### Questions
- There is no very detailed information about the base model, such as model parameters, VAE parameters, etc. Is the base model a private model?

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
This paper proposed a method to unify generative video super resolution framework to include multi-modalities, including text, images and videos. It helps to synthesize more details and achieve high fidelity outputs for various conditional inputs. Specifically, it first upscale and channel concatenate the low-res video latents with high-res latents. Then it treats the multi-ID images and reference videos as visual tokens and token-wise concatenate them with previous high-res latents. Such tokens are performed separate 2D self-attention by themselves and finally jointly performed 3D self-attention together with high-res tokens. For PE, it applies separate RoPE for each reference videos and the target video. The author also proposed SDEdit degradation which adds k steps noise to latents and decode back into RGB space and then add normal degradation.

### Strengths
- introduced additional information, including text, reference images, videos during video super resolution process, which makes the high-res detailed generated with guidance.
- Difficult-to-easy training order makes the multiple task performance better.
- Reference augmentation makes the output more robust.

### Weaknesses
- limited novelty of super resolution and reference-based generation task since LR channel concatenation and reference token-wise concatenation is typical in each domain of research. Using separate PE for reference tokens is also used in other papers.

### Questions
- The goal of SDEdit degradation is to mimic the real generated low-res video fidelity distribution. It would be helpful to has discussion about how close is it and propose any metrics to measure it.
- For individual RoPE assignment for multiple reference tokens, i.e., n_i to n_i + k_i, how are they selected? Does the order of reference content (images/videos) matter? Would the PE with smaller i biased during training?
- The output quality of base model might differs. How does the super resolution model handle difference quality inputs?

### Soundness
3

### Presentation
3

### Contribution
2
