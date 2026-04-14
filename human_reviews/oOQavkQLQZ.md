# FrameBridge: Improving Image-to-Video Generation with Bridge Models

- Decision: Reject
- Scores: 5, 3, 5, 3

## Abstract
Image-to-video (I2V) generation is gaining increasing attention with its wide application in video synthesis. Recently, diffusion-based I2V models have achieved remarkable progress given their novel design on network architecture, cascaded framework, and motion representation. However, restricted by their noise-to-data generation process, diffusion-based methods inevitably suffer the difficulty to generate video samples with both appearance consistency and temporal coherence from an uninformative Gaussian noise, which may limit their synthesis quality.
In this work, we present FrameBridge, taking the given static image as the prior of video target and establishing a tractable bridge model between them. By formulating I2V synthesis as a frames-to-frames generation task and modeling it with a \textit{data-to-data} process, we fully exploit the information in input image and facilitate the generative model to learn the image animation process.
In two popular settings of training I2V models, namely fine-tuning a pre-trained text-to-video (T2V) model or training from scratch, we further propose two techniques, SNR-Aligned Fine-tuning (SAF) and neural prior, which improve the fine-tuning efficiency of diffusion-based T2V models to FrameBridge and the synthesis quality of bridge-based I2V models respectively. 
Experiments conducted on WebVid-2M and UCF-101 demonstrate that: (1) our FrameBridge achieves superior I2V quality in comparison with the diffusion counterpart (zero-shot FVD 95 vs. 192 on MSR-VTT and non-zero-shot FVD 122 vs. 171 on UCF-101); (2) our proposed SAF and neural prior effectively enhance the ability of bridge-based I2V models in the scenarios of fine-tuning and training from scratch. Demo samples can be visited at: https://framebridgei2v.github.io/.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces an interesting image-to-video diffusion-based methods by leveraging the Denoising Diffusion Bridge Models (DDBM). It sets the initial state of denoising as temporally replicated first frame (i.e., static videos) and uses the DDBM to connect the distributions of the static videos and real videos sharing the same first frame. The reported results is better than naive diffusion models.

### Strengths
- The overall story is smooth and sounds interesting.

- The discussion on related works is sufficient.

- The experimental results seems to support the claim of contributions in the paper.

### Weaknesses
1. **I have a fundamental concern that I think the proposed method has no inherent difference between naive video diffusion models, especially considering the re-parameterization trick proposed in the paper.**

The paper says "Formulating a frame-to-frames generation task with a conditional noise-to-data sampling process, diffusion-based I2V systems suffer the difficulties to generate high-quality samples from uninformative Gaussian noise". This is to say that the authors believe the model can hardly learn to predict good videos $x_0$ from the mixture state of noise and data, that is $x_t = \alpha_t x_0 + \sigma_t \epsilon$.  Instead, providing the model the information from the statics images can help the model learn. The input for bridge models is formulated as $x_t = a_t x_0 + b_t \epsilon + c_t x_T$, where $x_T$ is the replicated initial frames.  This is also the reasonability of the bridge models.  

However, as proposed in Equation 7 of the paper, the authors remove all the information $x_T$ (The information of the initial frame is all removed), and the noisy state $x_t$ returns to the normal forms of standard diffusion models, which is the weighted sum of the target $x_0$ and the noise $\epsilon$.  Even though the authors claim that this operation helps them to adapt the standard diffusion models into the diffusion bridge models, I think this operation breaks the basic property of diffusion bridge models.

To this end, the only difference between the proposed method and the standard diffusion model is the prediction types. For standard DDPM, the model learns to predict the $\epsilon$. For the proposed method, the model learns to predict the $\epsilon + m_t x_T$. $m_t$ can be some time-dependent factors (not strict symbol form).   Since the model's input is the same now and the predictions just have an additional term $m_t x_T$ which is known at either training and inference stages, the influence of $x_T$ will be eliminated due to the sampling starting point difference between diffusion models and bridge models. I don't think the prediction type difference can make any difference. Therefore, I am in deep concern that the proposed method might have no intrinsic difference from naive video diffusion models. 

I am not very sure about my judgement though. I hope to propose the concern and inspire other experienced reviewers do further check on the  proposed technique.


2. **Connecting similar distributions is not uncommon.** Many works proposed similar ideas on image fields [1] [2] [3] for applications that they build a connection between the target image fields and the source image fields instead of simply using the source images as conditions. 

3.  **Detailed FVD learning curve for different baselines.** I would like the authors provide the detailed staged FVD learning curves (or other metrics) for different baselines. They can clearly show the learning speed of different methods and show performance gap of different methods trained with the same training budget.

4. In table 4, I would like to see the FVD for gaussian prior and Fn conditions. Since we can see from the table that, when both using Fn as conditions, the model have very similar performance with replicate prior and neural prior. This might indicate that the condition is more important than prior.

5. It is known that FVD has a very strong content bias and tends to misjudge the quality of video motion. Therefore I want the authors report the new-FVD proposed in the previous work. [4]
 


[1]  FMBoost: Boosting Latent Diffusion with Flow Matching. ECCV 2024 Oral

[2]  ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting. NeurIPS 2023, Spotlight, TPAMI@2024.

[3] Residual Denoising Diffusion Models. CVPR 2024.

[4] Content-Debiased FVD for Evaluating Video Generation Models.  CVPR 2024.

### Questions
I want the authors to show the training/inference pseudo code of their proposed method and standard video diffusion methods. It is the best way to provide a clear difference of these two methods.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
In this paper, the authors propose FrameBridge to achieve the image-to-video generation. Unlike previous diffusion-based methods, FrameBridge starts from a deterministic data point, the static image, and then maps it to the video latent through a bridge model. To achieve that, they further propose two techniques, SNR-
Aligned Fine-tuning (SAF) and neural prior.

### Strengths
1. Solving the distribution gap of training and inference is a reasonable motivation.
2. The video generated by FrameBridge is smoother with less mutation.
3. Compared to previous methods, FrameBridge achieves better metric scores on WebVid-2M and UCF-101.

### Weaknesses
1. Insufficient reference to related work. For example, “Common Diffusion Noise Schedules and Sample Steps are Flawed” proposes to align the SNR between the training stage and the inference stage. 
2. Some videos on the website can not be played. It seems that authors still modify the website when the review period starts. 
3. The video quality is not satisfactory. First, Maybe due to the size of the training set, the visual quality is not competitive (the worst one in my assigned papers). 
4. It seems that neural prior will limit the dynamics of the generated results. The transformation starting point of all frames comes from the transformation of the same frame. In this case, most generated results only contain small motions. The selected indicators are not helpful in judging the motion of the video. Alternatively, evaluating on a comprehensive benchmark like VBench will make the comparison more convincing.
5. Abstract: “modelling” -> “modeling”.

### Questions
Can you show the best ten cases from FrameBridge? In the field of video generation, the upper bound of the model is also a very important reference indicator.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a novel image-to-video (I2V) generation framework called FrameBridge, which models the frame-to-frames synthesis process with a data-to-data generative framework. Unlike traditional diffusion-based I2V methods, FrameBridge takes the given static image as the prior of the video target, establishing a bridge model between them. This approach reduces the burden of generative models and improves synthesis quality. The authors also propose two techniques, SNR-Aligned Fine-tuning (SAF) and neural prior, to further enhance the performance of FrameBridge.

### Strengths
1.	FrameBridge introduces a new data-to-data generation framework for I2V synthesis, which is different from traditional noise-to-data diffusion-based methods.
2.	By taking the given static image as the prior of the video target, FrameBridge reduces the burden of generative models and improves synthesis quality.
3.	The proposed SAF technique enables seamless knowledge transfer between pre-trained diffusion models and FrameBridge, improving fine-tuning efficiency.

### Weaknesses
1.	My primary concern about this paper is the experiment. I acknowledge that the bridge model may be a promising method for image-to-video diffusion models, and this paper has made an effort to fully utilize input image information. However, this paper only provides baselines for SEINE and DynamiCrafter, other works such as Animate Anyone [1] and Sparsectrl [2] have attempted to introduce an additional appearance encoding and fusion module to support more refined conditional image information utilization, which is an advantage over these methods. Therefore, this paper needs to provide a more detailed discussion. Moreover, ConsistI2V [3] designed FrameInit, which incorporates image priors into the initial noise, breaking the noise-to-data pattern and leading to better results. This paper does not provide a comprehensive comparison with other state-of-the-art I2V generation methods.
2.	Regarding evaluation metrics, FVD and IS are no longer reliable metrics. In fact, there are many benchmarks focused on video generation, such as T2V-CompBench [4] and VBench [5], which provide reliable evaluation metrics. The authors should use the most advanced benchmarks. Furthermore, this paper does not provide user studies.

[1] Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation, Hu et al., CVPR 2024
[2] Sparsectrl: Adding sparse controls to text-to-video diffusion models, Guo et al., ECCV 2024
[3] ConsistI2V: Enhancing Visual Consistency for Image-to-Video Generation, Re et al., TMLR 2024
[4] T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generationrk, Sun et al., arXiv 2024 
[5] VBench: Comprehensive Benchmark Suite for Video Generative Models, Huang et al., CVPR 2024

### Questions
Based on the weaknesses mentioned, this paper requires further revisions, therefore, I currently give it a score of 'marginally below the acceptance threshold'.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces FrameBridge, which formulates I2V synthesis as a frames-to-frames generation task rather than a conditional noise-to-frames generation. This approach allows the model to concentrate more effectively on learning image animation, enhancing both the temporal and appearance coherence of the generated video. Furthermore, to better utilize the priors from pre-trained T2V models, SNR Aligned Fine-tuning (SAF) and a neural prior are proposed to promote fine-tuning efficiency and improve the quality of generated videos. Extensive experiments demonstrate the effectiveness of FrameBridge in image-to-video generation tasks.

### Strengths
S1: Unlike previous diffusion-based methods, FrameBridge formulates I2V synthesis as an interesting frames-to-frames generation task rather than a conditional noise-to-frames generation.

S2: The paper presents good qualitative and quantitative results.

### Weaknesses
W1: In Table 1, some important baselines are missing, such as SVD [r1] and I2VGen-XL [r2], both of which are strong baselines and would be valuable for comparison. Additionally, I noticed in line 473 that DynamiCrafter was reproduced in this paper. I’m curious why the original weights couldn’t be used for testing, as the videos generated by DynamiCrafter in Figure 5 and on the project page seem to exhibit significant inconsistencies. This seems unusual, given that DynamiCrafter claims to rank first on the I2V benchmark of VBench [r3]. Clarifying the reproduction process and discussing potential discrepancies would be helpful. 

[r1] Blattmann, Andreas, et al. "Stable video diffusion: Scaling latent video diffusion models to large datasets." arXiv preprint arXiv:2311.15127 (2023).
[r2] Zhang, Shiwei, et al. "I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models." arXiv preprint arXiv:2311.04145 (2023).
[r3] Huang, Ziqi, et al. "Vbench: Comprehensive benchmark suite for video generative models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

W2: In Figure 5 and on the project page, most of the generated videos seem to exhibit only minor movements, which raises a concern that the proposed method may tend to produce videos with limited motion. Could the authors provide examples of the method generating videos with larger-scale actions, such as camera movements or vehicles in motion? Alternatively, discussing any limitations in generating more dynamic scenes would be valuable.

W3: In line 470 of the paper, it is noted that all video synthesis methods sample 250 steps, which is quite time-consuming. Since most models achieve good results with only 50 steps, could the authors clarify the justification for using 250 steps? Additionally, it would be helpful if the authors could provide a comparison of quality vs. number of steps, along with the inference time for different step counts.

W4: In Table 2, FrameBridge-NP is mentioned. Could the authors clarify whether the default FrameBridge includes a neural prior? Additionally, it would be helpful to specify if NP is used only in the context of training from scratch or if it is a default setting. If NP is only applicable or beneficial in specific training scenarios, clarifying these conditions would be useful.

W5: Table 2 should include experimental results of classic I2V network architectures, such as Dynamicrafter and I2VGEN-XL. Furthermore, I think it may be better to conduct ablation studies based on the setting in Table 1, as suggested in the Dynamicrafter paper, since generalizability is crucial for image-to-video generation models.

### Questions
I hope the authors can address the concerns raised in the "Weaknesses" section.

### Soundness
2

### Presentation
3

### Contribution
2
