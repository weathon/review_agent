# Consistent123: Improve Consistency for One Image to 3D Object Synthesis

- Decision: Reject
- Scores: 6, 5, 5, 6

## Abstract
Large image diffusion models enable novel view synthesis with high quality and excellent zero-shot capability. However, such models based on image-to-image translation have no guarantee of view consistency, limiting the performance for downstream tasks like 3D reconstruction and image-to-3D generation. To empower consistency, we propose Consistent123 to synthesize novel views simultaneously by incorporating additional cross-view attention layers and the shared self-attention mechanism. The proposed attention mechanism improves the interaction across all synthesized views, as well as the alignment between the condition view and novel views. In the sampling stage, such architecture supports simultaneously generating an arbitrary number of views while training at a fixed length. We also introduce a progressive classifier-free guidance strategy to achieve the trade-off between texture and geometry for synthesized object views. Qualitative and quantitative experiments show that Consistent123 outperforms baselines in view consistency by a large margin. Furthermore, we demonstrate a significant improvement of Consistent123 on varying downstream tasks, showing its great potential in the 3D generation field.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Consistent123 is an improved version of Zero123 by optimizing using extra cross-attention consistency with a progressive classifier-free guidance strategy. This cross-attention training also enables generating an arbitrary number of views during inference.

### Strengths
- A clear improvement in novel view synthesis. The video shared in the supplementary file clears shows the strength of Consistent123 over Zero123. 
- The cross-attention mechanism is simple yet effective.

### Weaknesses
1. Smoothness in Results. I noticed that the Consistent123 approach generally produces smoother results and seems to miss out on some finer details compared to Zero123. This observation is particularly evident in the first row of Fig. 4, and in the hat geometry depicted in Fig. 7: both in sections (a) bottom and (b) up. Have the authors identified potential strategies or modifications to address this shortcoming?

2. Concerns regarding arbitrary-length sampling. The methodology adopted uses a fixed number of views (8 views) during training. I'm concerned that this fixed view might adversely affect performance when dealing with arbitrary-length sampling at inference. This concern arises from a potential mismatch between training and test distributions. Any reason why using fixed number views during training? Is it only for simple implementation and faster training? An ablation study showcasing the performance with a random number of views as well during training would provide valuable insights and address this concern. 

3. Ablation on Zero123 pretraining. Could you present results when Consistent123 is trained from scratch without Zero123 pretraining?

### Questions
1. Presence of Ground Truth for clarity. I would recommend including the Ground Truth in Figs 1, 4, 5, and 6. Having a point of reference would greatly enhance the clarity and allow for a more informed evaluation of the results.

2. Visualization of cross attention during training. The manuscript currently lacks visualizations for cross-attention dynamics during training. It would be beneficial for readers to see how these cross-attention maps evolve and converge throughout the training process. 

3. It makes the paper stronger if you can show better 3D reconstruction results. For example, you can use your Consistent123 inside RealFusion [1] and Magic123 [2] to show state-of-the-art image-to-3D results. 

Other minor suggestions:
1. suggest to add more views in Fig 1 since there are empty space. You can also point out in the figure where Zero123 fails and you success to catch the audiences’ attention quickly. 
2. Show back views in Fig 7 (a) bottom and (b) up. 
3. Better to add cross attention between views in Fig.2 (a) as well, like a few red lines across views at the denoised views.

[1] Melas-Kyriazi, Luke, Iro Laina, Christian Rupprecht, and Andrea Vedaldi. "Realfusion: 360deg reconstruction of any object from a single image." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8446-8455. 2023.
[2] Qian, Guocheng, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee et al. "Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors." arXiv preprint arXiv:2306.17843 (2023).

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors aim to improve the view consistency for the novel view synthesis method based on image-to-image translation (i.e., Zero123). Specifically, They incorporate Zero123 with shared self-attention layers and additional cross-view attention layers. In addition, they propose a progressive classifier-free guidance strategy to balance the texture and geometry during the denoising process. Experimental results show that the proposed Consistent123 achieves better view consistency on multiple benchmarks compared to Zero123. The authors also demonstrate the potential of Consistent123 on various downstream tasks, such as 3D Reconstruction and image-to-3D generation.

### Strengths
1. The proposed method allows flexible view numbers compared to concurrent work MVDream. Experiments show that using arbitrary-length sampling with more view numbers could boost view consistency, indicating the effectiveness of the proposed method.
2. The proposed method is intuitive yet effective. By adding additional attention mechanisms, the authors improve the view consistency of Zero123.
3. The proposed progressive classifier-free guidance is interesting and alleviates the trade-off between geometry and texture.

### Weaknesses
1. The attention mechanisms are totally borrowed from previous work, such as shared self-attention from Cao et al. and cross-attention from video diffusion models.
2. For the shared self-attention layers, when the views are totally orthogonal, how will this shared self-attention act? Can this self-attention find correct correspondence? For example, in Figure 5 (right), when there is no shared self-attention, the resulting first view seems much more interesting. It would be better to have self-attention visualization in this case.
3. Considering Objaverse has 800K+ 3D models, the authors only picked up 100 objects for Table 1, which seems far from enough.
4. The proposed Consistent123 loads pretrained weight from Zero123 and fixes these weights. It would be fair to also have a version training from scratch.
5. Will the compromise solution introduce view inconsistency? It looks like there is no connection between the sampled views and the next round views.
6. The results on 3D reconstruction seem poor, where the results from Neus are blurry and low-quality.

### Questions
1. For the shared self-attention layers, is there any positional embedding? If not, does introducing a camera pose-aware positional embedding help? Do you have any insight on this? 
2. Since the current conditions on R and T are still geometry-free, I am worried that the proposed method's upper bound is limited.

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
The paper introduced a way that improves novel view synthesis model, e.g. zero123 by considering a multiview input in the diffusion model. While consider a shared self-attention machanism that all views  query the same key and value from the input view, which provides detailed spatial layout information for novel view synthesis.  In the method, It supports Arbitrary-length Sampling and adopted Progressive Classifier-free Guidance, yielding a further improvement of the synthesis. 

The resulting novel views looks more consistent than baseline. And from the supplimentary material.

### Strengths
1. The multiview input to the diffusion is good in achieving geometric consistency, comparing against zero123 base model. 

2. The design of progressive scheduler is interesting by jointly considering the benefit from large cfg vs small cfg, which leverage between texture and geometry. 

3. The paper demonstrates through qualitative and quantitative experiments that Consistent123 significantly outperforms baselines, zero123 in particular, in view consistency, showcasing substantial improvement in various downstream tasks.

### Weaknesses
Novelty is clear, while there are several publications available with open-sourced papers. Such as magic123,  zero123 xl, sync-dreamer.   for synthesizing new views and do the 3D reconstruction using SDS or direct pixel loss based on NeuS. Wonder the author may compare the results with the opensourced recon-models. 

From the experimental results after 3D reconstruction. It looks like a black biased back side are generated. Which in my perspective, they are no better than the pulished methods such as that has been implemented in threestudio [url: https://github.com/threestudio-project/threestudio] [which is available before the submission].  The autho may explain why the reconstructed results

### Questions
1. The generated views are still not fully consistent before the 3D model is reocnstructed, while the renderred image after reconstruction looks much worse.  is there any thoughts in further improve the consistency so the quality gap between diffused output and render-view can be minimized ? 

2. How it generalizes towards more sophisticated images ?  Please also provide some faliure cases for a better understand of the limitations.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a novel method to synthesize a set of images of any objects from novel view given a single image as input. One of many challenges in this task is to generate consistent images in terms of geometry and appearance. To this end, the authors propose to generate multiple images simultaneously and enable cross-attention between novel images at different viewpoints. To strike a balance between geometry and texture of generated images, they also propose a progressive Classifier Free Guidance (CFG) after observing that a larger CFG often leads to better geometry but poor texture and a smaller CFG causes an opposite result. Experiments demonstrate that the proposed method outperform a popular baseline,  Zero123, on image synthesize.

### Strengths
- Consistency in novel view synthesis is at the core of many image-to-3D task. The proposed method is able to outperform Zero123 by a large margin qualitatively and quantitatively.
- The observation of more generated views improving consistency Is useful to other image- or text-conditioned novel view synthesis works.
- Paper is generally easy to follow

### Weaknesses
- V_c should be after softmax in Eq. 5.
- The name "shared self-attention" is confusing to me. It in fact is a cross-attention from the novel views to the input view. Why is it called self-attention?
- Only qualitative results were presented for image-to-3D tasks.

### Questions
- How many views were used in the NeuS experiment?

- No texture on the spray bottle in Figure 7?

- Is the Super Mario a failure case since the object has flattened? Could it be related to the progressive CFG? Figure 3 seems to suggest that a large CFG could lead to flat objects.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair
