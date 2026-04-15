# Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5

## Abstract
Significant advancements have been achieved in the realm of large-scale pre-trained text-to-video Diffusion Models (VDMs). However, previous methods either rely solely on pixel-based VDMs, which come with high computational costs, or on latent-based VDMs, which often struggle with precise text-video alignment. In this paper, we are the first to propose a hybrid model, dubbed as MPL-Video , which marries pixel-based and latent-based VDMs for text-to-video generation. Our model first uses pixel-based VDMs to produce a low-resolution video of strong text-video correlation. After that, we propose a novel expert translation method that employs the latent-based VDMs to further upsample the low-resolution video to high resolution. Compared to latent VDMs, MPL-Video can produce high-quality videos of precise text-video alignment; Compared to pixel VDMs, MPL-Video is much more efficient  (GPU memory usage during inference is 15G vs 72G). We also validate our model on standard video generation benchmarks. Our code will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces MPL-Video, a hybrid model for text-to-video generation that combines pixel-based and latent-based Video Diffusion Models (VDMs). MPL-Video overcomes limitations of previous methods, e.g., precise text-to-video alignment, by initially employing pixel-based VDMs to generate a low-resolution video with strong text-video correlation. Subsequently, a uniquely proposed expert translation method leveraging latent-based VDMs is applied to enhance the low-resolution output to high resolution. MPL-Video demonstrates superior performance compared to its pixel-based and latent-based counterparts and has significantly lower GPU memory usage during inference (15G as opposed to 72G).

### Strengths
1. The paper introduces an approach by combining pixel-based and latent-based VDMs, overcoming the individual limitations of each. This hybrid model, MPL-Video, employs pixel-based VDMs for keyframe modules and temporal interpolation at low resolution, ensuring precise text-video alignment and natural motion with reduced computational costs.

2. The proposed method ensures enhanced super-resolution of videos, achieving precise text-to-video alignments with optimized computational costs, thereby advancing the field by offering a balanced solution that does not compromise quality for computational efficiency.

### Weaknesses
1. The assertion that
> Pixel-based VDMs can generate motion accurately aligned with the textual prompt

appears somewhat unsubstantiated. The manuscript could be fortified by providing more compelling empirical evidence or a thorough rationale that unequivocally corroborates this claim, thereby enhancing the reliability of the proposed methodology.

2.  Section 3.3 deliberates the choice of pixel diffusion over latent diffusion. However, the justification proffered seems to lack cogency. The methodology initiates with a smaller resolution in the pixel space, transitioning later to a higher resolution in the latent space. A more elaborate elucidation regarding this choice would be instrumental. Specifically, it would be insightful to comprehend why a high resolution is not directly employed in conjunction with latent diffusion, elucidating the inherent advantages of beginning with a lower resolution in the pixel space.

3. The manuscript primarily focuses on comparing GPU memory savings as a metric for evaluation. While this is undoubtedly a vital aspect, a more holistic approach considering additional pivotal parameters such as inference latency would substantially enrich the comparative analysis.

### Questions
1.Could the authors please clarify whether the visualization results presented in Figure 2 were generated using identical random seeds and training data? Based on prior experience, utilizing consistent random seeds and training datasets typically results in latent-based VDMs exhibiting similar patterns at both low and high resolutions. However, the patterns depicted in Figure 2 seem to considerably diverge from this expectation. A clarification regarding this discrepancy would be highly appreciated.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to combine the pixel-based and latent-based diffusion models under the cascaded framework for text-to-video generation. They use three pixel-based modules to generate videos of size 29 × 256 × 160 and an additional latent-based module to generate videos of size 29 × 576 × 320. The motivation is that the pixel-based models provide better text-video alignment and the latent-based models require less computation and memory.

### Strengths
The idea of marrying pixel-based and latent-based diffusion models is clearly presented and relatively novel. The paper is generally well-written and easy to follow. The resulting model achieves competitive quantitative results on standard benchmarks and only requires 15GB memory during the inference. The qualitative results shown on the anonymous website are appealing and promising. The authors promise to release the code and models.

### Weaknesses
1. Motivation: I found that the argument that the latent-based model produces less text-aligned results than the pixel-based model is not very convincing since we do observe a similar level of text-video alignment in Stable Diffusion or Video LDM compared with other pixel-based diffusion models. If the reason is that the latent is too small (8×5 for 64×40 videos), then could the authors explain what might be the benefits of generating such a low-res video in the first stage? Also, one can always use a smaller compression rate if the latent needs to be larger, since a larger compression rate is only needed to reduce more computation and memory.

2. Motivation: the other motivation of the proposed method is that the latent-based model can save memory when upsampling in the high-resolution. However, pixel-based models like Make-A-Video often mitigate this issue by reducing the number of parameters in the latter upsampling modules. Thia makes sense as the upsampling task is much simpler than the generation task. Therefore, this motivation is also less convincing to me. Specifically, I wonder how is the 72GB memory calculated in Table 4.

3. Visual result: in the comparison results with Make-A-Video in Figure 5, it seems that the video generated by MPL-video has much less motion. 

4. There are several unclear details in the paper. Please see the questions below.

### Questions
1. In Section 3.2, how does the temporal attention work exactly? Are the spatial dimensions transposed to the batch dimension, or is the attention applied to both the spatial and temporal dimensions?

2. In Section 3.5, when generating the first segment, what is used as the conditioned high-resolution frame?

3. In Section 4.1, are ModelScope model weights used to initialize the last super-resolution model?

4. The WebVid-10M dataset is known to have plenty of watermarks. Why is this not seen in the generation results?

5. In Table 4, is there a typo in the last row, which should be pixel-based for the low-res stage and latent-based for the high-res stage? If that is the case, why is the CLIPSIM score different from the one in Table 2?

### Soundness
3 good

### Presentation
3 good

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
This paper focus on text-to-video generation by exploring the combination of pixel-based and latent-based video diffusion models (VDMs). The authors found that pixel-based VDMs can generate accurate motion described by the text but are expensive. Latent-based VDMs are more resource-efficient but cannot cover rich visual semantic details. Therefore, the authors propose a hybrid model, which generates low-resolution videos of strong text-video correlation with pixel-based VDMs and upsamples them to high resolution with latent-based VDMs.

### Strengths
The motivation of mixing these two models is also intuitive. The authors conduct a comprehensive comparison between pixel-based and latent-based VDMs. They also explained how they choose between pixel diffusion and latent diffusion in different modules in their method part.

Experiments demonstrate the superiority of their proposed method in terms of its memory cost and performance.

### Weaknesses
1. Since there are three models involved in inference (pixel-based keyframe generation model, temporal interpolation model and 2 super-resolution models), the time cost of this method could be high.

2. The overall pipeline is not something new. It first generates keyframes and then temporally interpolates between keyframes and upsamples to higher spatial resolution. This pipeline has been proposed in previous approaches, such as make-a-video and Imagen Video and NUWA-XL[1]. The authors should also cite [1] as a related work.

3. Typos:

* $T$ is defined as the number of diffusion steps in 3.1 and it is also defined as the number of frames in 3.2.

* From my understanding, $\{z^{i}_t|t=1…T\}$ are equal. If that is true, $z^{i}$ and $z^{i+1}$ in sec. 3.4 should keep consistent with $z^{i}_t$ and $z^{i+1}_t$ in Figure 4 (b).

* C in Section 3.4 should be in the math format $C$

* “.,” in formula (6)

* In Table 2. “11.09” should be bolded instead of “13.17”, which means MPL-Video does not achieve the best performance in FID-vid as claimed in Sec. 4.2.

[1] Yin et al. NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation. ACL 2023

### Questions
1. In Table 4, how different combinations of pixel-based and latent-based VDMs perform in terms of video fidelity?

2. How does the time cost of your proposed method compare to that of previous methods?

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
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposed a new text-to-video generation model based on cascaded diffusion models, where the first text-to-video model, the temporal interpolation, and low-res SR models are based on pixel diffusion models. On the contrary, the last high-res SR model is based on the latent diffusion model, which is the main difference between the proposed method with existing baselines such as the Make-A-Video and Imagen Video, which are fully based on pixel diffusion models. Experimental results on UCF101 and MSRVTT show that the proposed method has comparable video quality while the text-following ability is improved.

### Strengths
- [writing] This paper is easy to follow for me, and the overall writing is very good.
- [method] The idea of combining the pixel space diffusion model and the latent space model to have both the benefits of good text following and computation saving is interesting.
- [experiment] The experimental results show that the proposed method has comparable or better video generation quality in terms of the FVD and CLIPSIM while the memory spend has been reduced.

### Weaknesses
- [writing] Should the last row of Table 4 be pixel-based then latent-based? Typos: "a expert".
- [method] The main weakness of the proposed method is that it is very similar to existing works like Make-A-Video in the algorithmic design, where the main difference lies in the choice of latent- or pixel-based diffusion models for different functions. Existing works (like those compared in the experiment section) have already demonstrated that both pixel and latent diffusion models can work well on text-to-video generation. I am unsure whether the combinational use can bring new insight into the community.
- [method] In the expert translation part, the authors mentioned that training the latent diffusion model only on 0-900 time steps instead of 0-1000 yields better video quality. Is there any quantitative evidence about the quality improvement?
- [method] I find the CLIPSIM improvement of the proposed method is insignificant while one of the main benefits of using the pixel-space diffusion model is that it can preserve better semantics and appearances. Are there any other metrics that can demonstrate the benefit of using a pixel-space diffusion model over latent-space ones?
- [experiment] I noticed that the Imagen Video and Make-A-Video's papers mentioned that their models have a parameter size of 16.3 and 9.7 billion, respectively. I am unsure how large is the model size of the proposed method. How to compare the above parameter size with the the max memory shown in Table 4?

### Questions
Please see the weakness part.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
