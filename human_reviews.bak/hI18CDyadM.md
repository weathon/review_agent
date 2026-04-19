# Adaptive Window Pruning for Efficient Local Motion Deblurring

- Decision: Accept (poster)
- Scores: 8, 6, 3

## Abstract
Local motion blur commonly occurs in real-world photography due to the mixing between moving objects and stationary backgrounds during exposure. Existing image deblurring methods predominantly focus on global deblurring, inadvertently affecting the sharpness of backgrounds in locally blurred images and wasting unnecessary computation on sharp pixels, especially for high-resolution images.
This paper aims to adaptively and efficiently restore high-resolution locally blurred images. We propose a local motion deblurring vision Transformer (LMD-ViT) built on adaptive window pruning Transformer blocks (AdaWPT). To focus deblurring on local regions and reduce computation, AdaWPT prunes unnecessary windows, only allowing the active windows to be involved in the deblurring processes. The pruning operation relies on the blurriness confidence predicted by a confidence predictor that is trained end-to-end using a reconstruction loss with Gumbel-Softmax re-parameterization and a pruning loss guided by annotated blur masks. Our method removes local motion blur effectively without distorting sharp regions, demonstrated by its exceptional perceptual and quantitative improvements (+0.28dB) compared to state-of-the-art methods. In addition, our approach substantially reduces FLOPs by 66% and achieves more than a twofold increase in inference speed compared to Transformer-based deblurring methods. We will make our code and annotated blur masks publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a local motion deblurring Transformer with adaptive window pruning, which only deblur the active (blurry) windows. The windows are classified as active/inactive according to the predicted bluriness confidence score.

### Strengths
Overall, I think this paper has a novel idea and achieves good results in terms of performance and efficiency. I tend to accept this paper due to following reasons.

1. adaptive window pruning that saves computation on unnecessary attention windows.

2. bluriness confidence prediction that works well for both local motion prediction and global motion prediction

3. annotated local blur masks on ReLoBlur

4. well-designed experiments, well-presented figures and well-written paper.

### Weaknesses
Below are some concerns and suggestions.

1. Since the confidence predictor only uses MLP layers. How many pixels did you shift? Is the feature shift necessary to enlarge the receptive field of the neighbourhood? 

2. What is the mask prediction accuracy on validation set?

3. How did you decide the border when annotating the masks for blurry moving objects?

4. If a patch is always abandoned, how is it processed? What layers it will be passed into during inference?

5. It could be better to provide the results of two special cases (masks are all-ones/all-zeros) in the tables as a reference.

6. Why did you only report real-world results on large images? Do you have a chart for comparison on PSNR/FLOPs/runtime under different image resolutions.

7. The key of this method is the adaptive window pruning. It is better to provide an ablation study with the rest tricks as a baseline (i.e., no window pruning in training and testing).


Minor:

1. Unfinished paragraph in page 2.

2. For figure 1, I think it might be better to visualise the attention window borders, for example, by adding solid lines to show the windows.

3. The summarised contributions are a bit overlapped (point 1 and point 2). I think it's better to claim adaptive window pruning and bluriness confidence prediction as two contributions.

4. I think there is no need to distinguish between AdaWPT-F and AdaWPT-P. Just be simple and united. The name of AdaWPT is enough.

5. Comparison in Figure 6 is not visible.

6. Is there an example on a globally clear image (e.g., a still scene). Contrary to Figure 7, will the decision map be all zeros?

7. Are there other similar works in image restoration/deblurring? Are there some connections between this method and some "blur kernel prediction + restoration" methods (e.g. Mutual Affine Network for Spatially Variant Kernel Estimation in Blind Image Super-Resolution)?

### Questions
See weakness.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The presented work delves into an interesting research problem: local motion deblurring. It introduces a novel approach known as LMD-ViT, constructed using "Adaptive Window Pruning Transformer blocks (AdaWPT)." AdaWPT selectively prunes unnecessary windows, focusing powerful yet computationally intensive Transformer operations solely on essential windows. This strategy not only achieves effective deblurring but also preserves the sharp regions, preventing distortion. Moreover, their method significantly accelerates inference speed. Additionally, they have provided annotated blur masks for the ReLoBlur dataset. The utility of this approach is showcased on both local and global datasets, where it demonstrates substantial performance improvements on local motion deblurring (the ReLoBlur dataset) and competes favorably with baseline methods in the realm of global deblurring.

### Strengths
1.	This paper addressed problems of single image local motion deblurring, which is very essential in today’s photography industry. The presented method is the first to apply sparse ViT in single image deblurring and may inspire the community to enhance image quality locally.
2.	The proposed pruning strategy including the supervised confidence predictor, the differential decision layer and pruning losses are reasonable and practical. It combines window pruning strategy with Transformer layers, only allowing blurred regions to go through deblurring operations,  resulting in not only proficient deblurring but also the preservation of sharp image regions.
3.	The quantitative and perceptual deblurring performances are obvious compared to baseline methods.
4.	The presented method derives a balance between local motion deblurring performance and inference speed, as shown in the ablation study and experiments. The proposed method reduced FLOPs and the inference time largely without deblurring performances dropping on local deblurring data.
5.	The authors provided annotated blur masks for the ReLoBlur dataset, enhancing the resources available to the research community.

### Weaknesses
1.	The authors did not mention whether the presented method LMD-ViT requires blur mask annotation during inference. This is crucial because if the method does require blur masks before inference, it would be helpful to provide instructions on how to generate them beforehand and assess their practicality.
2.	The proposed method uses Gumble-Softmax as the decision layer in training and Softmax in inference. The equivalence of the two techniques in training and inference is not discussed.
3.	In the user study experiment, the absence of an explanation regarding the camera equipment used is notable. This is important because when images from the same camera share a common source, the blurriness often exhibits a consistent pattern. Therefore, including images from the same camera would allow us to assess the proposed method's robustness.
4.	Some references are missing, like “Window-based multi-head self-attention” in page 2, and “LeFF” in Section 2.4.1.

### Questions
please refer to the weaknesses.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper attempts to tackle local motion deblurring. Existing deblurring literature mostly focuses on general deblurring (without specifying global or local regions). This could be a disadvantage in computation and generalization in scenarios where only a small part of a high-resolution image is blurred. Therefore, this work proposes a transformer-based network called LMD-ViT. The authors make use of adaptive window pruning and blurriness confidence predictor in the transformer blocks to ensure the network focuses on local regions with blurs. Quantitative and qualitative results are presented on the ReLoBlur and GoPro datasets. The effectiveness of different design choices is analyzed.

### Strengths
An adaptive window pruning strategy is adopted to focus the network computation on localized regions affected by blur and speed up the Transformer layers.

A carefully annotated local blur mask is proposed for the ReLoBlur dataset to improve the performance of local deblurring methods.

### Weaknesses
The organization of the paper can be improved.

1) The methodology (Sec. 2) consists of too many (unnecessary) acronyms. Moreover, there are some inconsistencies when citing previous works (for example, LBAG (Li et al., 2023), LBFMG (Li et al., 2023), etc.). It would be better for the submission would strongly benefit from polishing the writing.

The settings of the experiments need more explanation.

2) It is not clear why the GoPro dataset is used for training along with the ReLoBlur training set. In previous works, such as LBAG, only the ReLoBlur dataset is used (see Table 4).

The novelty of the submission needs to be clarified.

3) It would be better to discuss the differences between LBAG and Uformer. Compared to LBAG, it simply substitutes the CNN architecture with a Transformer. All the other modules, including sparse ViT, W-MSA (Window-based multi-head self-attention), and LeFF (locally- enhanced feed-forward layer), have been introduced in previous deblurring works.

The fairness of the experiments.

4) The transformer baselines, such as Restormer and Uformer-B, are not trained with the local blur masks, which are deployed during their training by the proposed methods. This makes the comparison in Table 1 unfair.

Unclear parts.

5) Please explain in detail how the authors manually annotate the blur masks for the ReLoBlur dataset.

6) The baseline results reported in Table 1 are higher than those in their original papers, e.g., LBAG. It would be better to give the reasons and more details when introducing Table 1. The baseline results reported in Table 1 are higher than those in their original papers, e.g., LBAG for 34.85 dB.

Typo: 
Table table 4 in Appendix C.
Decision map in Figure 7.

### Questions
See Weaknesses.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
