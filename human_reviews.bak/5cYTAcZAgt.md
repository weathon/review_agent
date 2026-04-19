# SAN-Diff: Structure-aware noise for super-resolution diffusion model

- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Recent advances in diffusion models, like Stable Diffusion, have been shown to significantly improve performance in image super-resolution (SR) tasks. However, existing diffusion techniques often sample noise from just one distribution, which limits their effectiveness when dealing with complex scenes or intricate textures in different semantic areas. With the advent of the segment anything model (SAM), it has become possible to create highly detailed region masks that can improve the recovery of fine details in diffusion SR models. Despite this, incorporating SAM directly into SR models significantly increases computational demands. In this paper, we propose the SAN-Diff model, which can utilize the fine-grained structure information from SAM in the process of sampling noise to improve the image quality without additional computational cost during inference. In the process of training, we encode structural position information into the segmentation mask from SAM. Then the encoded mask is integrated into the forward diffusion process by modulating it to the sampled noise. This adjustment allows us to independently adapt the noise mean within each corresponding segmentation area. The diffusion model is trained to estimate this modulated noise. Crucially, our proposed framework does NOT change the reverse diffusion process and does NOT require SAM at inference. Experimental results demonstrate the effectiveness of our proposed method, which exhibits the fewest artifacts compared to other generated models, and surpassing existing diffusion-based methods by 0.74 dB at
the maximum in terms of PSNR on DIV2K dataset.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper proposes a new model, SAN-Diff, to enhance the performance of image super-resolution (SR) using diffusion models. By leveraging the Segment Anything Model (SAM), SAN-Diff incorporates fine-grained structural information into the noise sampling process, thereby improving the recovery of fine details in images. During training, the model encodes structural position information from SAM into the segmentation mask, which is then used to modulate the sampled noise. The proposed framework integrates this structural information without increasing computational costs during inference.

### Strengths
1. The preliminary section is commendably clear and comprehensive, effectively setting the stage for the subsequent content.
2. This paper excels in its theoretical foundation, for providing a meticulous and detailed explanation of the adjusted forward and reverse diffusion processes.
3. The combination of the renowned Segment Anything Model (SAM) and popular diffusion models is particularly interesting.

### Weaknesses
1. Almost all of the quotes in the article are in the wrong format. Please refer to the difference between `\citet` and `\citep`.
2. For the super-resolution task, it is suggested to provide general experiments with multiple scales, such as $\times 8 $.
3. Only SRDiff is compared in the Table 1. NOT various diffusion-based image super-resolution methods.

### Questions
1. Why not compare the proposed method with common baselines like EDSR and other widely used models? 
2. MANIQA, which is a metric for no-reference image quality assessment, seems less common for this super-resolution task. Why not use the more popular LPIPS metric?
3. Since SAM is not used during the inference, so how to utilize the fine-grained structure information from SAM in the inference?
4. To the best of our knowledge, generative models in the super-resolution often obtain lower performance on objective metrics (such as PSNR, SSIM) but higher on subjective ones. Therefore, the higher PSNR is not persuasive.
5. Results in Table 2 (such as SPSR and those of other baselines) are significantly lower than those reported in their original articles. It seems very strange.
6. Some diffusion models like Stable Diffusion (mentioned in the abstract) or ControlNet can also inject structure information (like semantic map) into the diffusion process. Why not compare the results of these models?

### Soundness
2

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
3

### Summary
The authors proposed SAN-Diff for image super-resolution task. The authors noticed that one potential limitation of existing works could be sampling from one distribution during the diffusion process, and might hurt the generated image results when the scene is complex. They proposed to using a positional embedding, calculated with respect to regions segmented by SAM, as the condition of an existing work Srdiff, and achieved good performance, on multiple datasets.

### Strengths
The authors’ proposal sound valid as the spatially uniform sampling could be a place where the potential improvement can happen. The math deduction looks valid. The computation cost caused by the proposed design seems neglectable, in comparison with the original approach. Experimentations of different model comparisons and ablation studies are comprehensive.

### Weaknesses
It seems like the authors have a strong assumption that the SAM always provides accurate segmentation results. Although it is a powerful segmentation model could be used to zero-shot inference on everyday object, it is “statistically” strong. Therefore, your experimentations also show quantitatively “statistically” better. It would be great if the authors could discuss about the limitations of using SAM as the segmentation mask provider.

### Questions
1. For the example shown in Fig 5,6, is it possible for the authors provide the segmentation masks using SAM? Then, it will help us to see if the segmentation mask provided by SAM matches the “better” parts of the SR images you generated, and justify your conclusions.
2. In Figure 1, it is hard for me to think of why this improvement happens because of the contribution from SAM. I highly doubt that the image region you contoured can be identified as different “regions” segmented by SAM. How would the authors justify the contribution of SAM?
3. In Table 2, I see the proposed model performed the best. However, the FID score for SRDiff and SAN-Diff suddenly dropped a lot across different datasets (the last two rows), which causes my concern that if the authors didn’t choose good enough baselines. Please justify. I also would like to refer other reviewers’ comments.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes SAN-Diff, a structure-aware noise modulation method for super-resolution (SR) diffusion models. SAN-Diff uses the Segment Anything Model (SAM) to generate fine-grained segmentation masks, which guide the noise modulation process, allowing the diffusion model to apply distinct noise distributions to different areas of the image. The Structural Position Encoding (SPE) module is introduced to integrate position information into these masks.

### Strengths
1. By modulating noise based on segmentation areas, SAN-Diff enhances the preservation of local structures, particularly useful for complex scenes and intricate textures.
2. SAN-Diff uses SAM-generated masks only during training, avoiding computational burdens during inference while retaining structure-level detail restoration.

### Weaknesses
1.  Lack of Novelty in Methodology: The SAN-Diff approach heavily relies on existing methods, including the Segment Anything Model (SAM) for segmentation and structural positional encoding (SPE) techniques. This reliance limits the novelty of the proposed methodology, without introducing fundamentally new innovations tailored for super-resolution tasks.
2. Limited Novelty in Designing Loss Function: The loss function shown in Eq.5 mainly follows a standard existing MSE approach focused on noise prediction. Only the structurally positioned embedded mask is just added  to it.
3. Limited Exploration of Alternative Modulation Strategies: SAN-Diff exclusively uses a segmentation-mask-driven approach for noise modulation without investigating other methods, such as feature-based techniques. 
4. No Discussion on the Time Complexity of SPE: The paper mentions negligible cost but does not clarify the computational time taken by  SPE module for generating masks before training. 
5. Dependency on High-Resolution Training Data: The method requires SAM-generated segmentation masks for all high-resolution (HR) training images, which may not always be feasible in datasets lacking extensive HR samples or where SAM’s segmentation quality is inconsistent. This reliance could restrict SAN-Diff’s application in low-data or low-resolution scenarios.

### Questions
- Could the authors mention if there are any unique aspects in how SAM and SPE are integrated or applied that are tailored for this task?
- Could the authors try using the other losses like structural loss instead of standard MSE loss for better results?
- Could the authors mention if they tried alternative feature extraction techniques like -depth maps, CNN features etc. instead of segmentation masks.
- Could the authors provide concrete timing measurements or complexity analysis for the SPE module. This would help to understand the practical implications of using this approach.

### Soundness
3

### Presentation
3

### Contribution
3
