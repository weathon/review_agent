# Revisiting Vector-Quantization for Blind Image Restoration

- Decision: Reject
- Scores: 6, 8, 3, 3

## Abstract
Vector-Quantization (VQ) generative models are widely used to learn a high-quality (HQ) codebook and a decoder as powerful generative priors for blind image restoration (BIR). In this paper, we revisit the key VQ process in VQ-based BIR methods, and provide three close observations on the side effects of VQ for code index prediction: 1) confining the representational capability of HQ codebook, 2) being error-prone on code index prediction, and 3) under-valuing the low-quality (LQ) feature for BIR. These observations motivate us to replace discrete VQ selection by continuous feature transformation from input LQ image to output HQ image with the HQ codebook. To this end, in this paper, we propose a new Self-in-Cross-Attention (SinCA) module to augment the HQ codebook with the LQ feature of input LQ image and perform cross-attention between LQ feature and input-augmented codebook. In this way, our SinCA extends the representational capability of the HQ codebook and effectively leverages the self-expressiveness property of input LQ image. Experiments on four typical VQ-based BIR methods demonstrate that, by replacing the VQ process with transformers using our SinCA, they achieve better quantitative and qualitative performance on blind image super-resolution and blind face restoration. The code will be publicly released.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores the limitations of using Vector Quantization in blind image restoration tasks and highlights three key challenges: restricted representational capacity, prediction errors in code index selection, and undervaluing low-quality image features. To address these limitations, the authors propose a Self-in-Cross-Attention module, which augments the high-quality codebook by performing cross-attention between the HQ codebook and the LQ feature map. This module enables a continuous feature transformation process rather than relying on traditional discrete VQ, thereby enhancing the expressive power of HQ representations and improving restoration accuracy. Experimental results show that integrating SinCA into existing VQ-based BIR methods significantly improves performance on super-resolution and face restoration tasks.

### Strengths
+ It is interesting to see that the authors give a deep analysis of using Vector Quantization in blind image restoration tasks. The paper provides both quantitative metrics and visualizations to support the need for their proposed method.
+ The introduction of SinCA is a compelling solution to overcome VQ's limitations in blind restoration tasks by shifting from discrete selection to continuous feature transformation, expanding the capacity for HQ representation.
+ SinCA shows measurable improvement in image quality restoration, achieving higher metrics across various datasets for super-resolution and face restoration compared to traditional VQ-based methods.
+ The model is tested across multiple BIR benchmarks and configurations, demonstrating the broad applicability and effectiveness of the SinCA module in various restoration contexts.

### Weaknesses
- The additional complexity introduced by cross-attention mechanisms may lead to increased computational costs, which might limit its applicability in real-time or resource-constrained environments.
- The reliance on the HQ codebook for SinCA could lead to overfitting, especially if the training data for the HQ codebook does not fully represent the diversity of LQ images encountered in real-world settings.

### Questions
- In comparison with other methods, how does the introduction of the Self-in-Cross-Attention (SinCA) module affect computational complexity?
- In the quantitative comparison, the retrained methods have an obviously inferior performance than the released models. I wonder whether the authors did not re-train their models carefully.
- The degradation usually has some bias towards some specific degradation. In the visual comparison, it is better to provide more results on real-world LR images, rather than these used synthetic data.
- How does SinCA scale with larger or more complex codebooks, especially for high-resolution images? Does the performance or efficiency significantly change with codebook size, and how do the authors ensure the balance between restoration quality and computational load?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this article, the authors rethink the key VQ process in VQ-based BIR methods and, based on observations, raised three issues: 1. VQ limits the representational capacity of HQ code books; 2. VQ is prone to errors in code index prediction; 3. VQ underestimates the importance of low-quality features in BIR. To address these issues, the author proposed using continuous features to replace discrete VQ selection and introduced a novel Self-Cross Attention module (SinCA). This module enhances the features of HQ code books and LQ images and performs cross-attention between LQ features and the enhanced code books. The authors conduct extensive experiments on various datasets to demonstrate the effectiveness of the SinCA module in different blind super-resolution tasks, showing that this method can achieve better quantitative metrics and visual performance.

### Strengths
Advantages: 1.This paper proposes a novel Self-Cross Attention module and the use of continuous features to replace discrete VQ selection, providing a different perspective for subsequent blind super-resolution research. 2.The observations made in this paper are quite reasonable and the arguments are well-substantiated. 3.The writing of this paper is fluent and generally conforms to academic writing standards.

### Weaknesses
On page 4, lines 174-178, the paper introduces the overall loss function, which is composed of reconstruction loss, code book loss, and commitment loss. However, the paper seems to lack a study of the loss function. If the paper could supplement this aspect, it would make the argument more complete.
On page 5, in Figure 3, the connection between the images and the text appears not to be very tight. If this issue could be improved, it would make the logical argumentation of the paper more fluid.
In the experimental part of the paper, the author conducted numerous experiments on different datasets. However, since it is in the field of blind super-resolution, if experiments on image recovery under adverse conditions could be added, it would make the experiments in the paper more comprehensive.
Conclusion: If the aforementioned shortcomings could be addressed, this article should be accepted.

### Questions
On page 4, lines 174-178, the paper introduces the overall loss function, which is composed of reconstruction loss, code book loss, and commitment loss. However, the paper seems to lack a study of the loss function. If the paper could supplement this aspect, it would make the argument more complete.
On page 5, in Figure 3, the connection between the images and the text appears not to be very tight. If this issue could be improved, it would make the logical argumentation of the paper more fluid.
In the experimental part of the paper, the author conducted numerous experiments on different datasets. However, since it is in the field of blind super-resolution, if experiments on image recovery under adverse conditions could be added, it would make the experiments in the paper more comprehensive.
Conclusion: If the aforementioned shortcomings could be addressed, this article should be accepted.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes to span the discrete HQ codebook to continuous space and incorporate the LQ feature for VQ-based blind image restoration (BIR). Building upon the three side effects of VQ for code index prediction, the SinCA module is proposed. Experiments show the effectiveness of the proposed method.

### Strengths
1. This paper provides detailed analysis of the side effects of VQ-based code index prediction for BIR, including confined codebook representation capability, index prediction error, and LQ feature under exploitation.
2. Experiments show the effectiveness of the proposed SinCA module for blind image restoration.

### Weaknesses
1. This paper aims to point that the HQ codebook alone is insufficient for blind image restoration, however, many previous methods have proposed to utilize the LQ feature for quality and fidelity tradeoff, such as [CodeFormer; NeurIPS, 2022], it seems like we couldn't  accomplish the same balance in proposed SinCA framework currently, and the final result is totally deterministic and uncontrolled.
2. The paper argue that the representation capability of the discrete codebook is confined, is there any experiments to validate that when we span the codebook space to continuous space through the attention operation, could we obtain any benefit compared to the discrete code index? And I mean no LQ feature involved here.
3. The mutual attention adopted in SinCA is heavily utilized in literature, such as reference-based image/video manipulation, which may insufficient to support the contribution of the paper.
4. The illustration and notation of Fig. 3 is unclear, and should be rephrased.

### Questions
1. It is unclear what the different color mean in Fig. 3(a), and why multiple codebook marks interleaved with LQ feature.
2. In sec. 4.2, is that reasonable to formulate the GT indices for transformer prediction with nearest-neighbor feature matching, then why we need transformer, and the consequent conclusion are doubted.
3. It is suggested to provide a deeper analysis why Fig. 2(c) achieves extreme lower code prediction accuracy than Fig. 2(b), considering the codebook usage result in Fig. 2(a).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper theoretically analyzes the shortcomings of recent VQ algorithm framework in BIR tasks, and aims to illustrate that the one-hot form of code matching will cause limitations in the expressive ability of the codebook, a high matching error rate, and underestimation of the role of low-quality features, resulting in a decrease in recovery performance. Therefore, the paper proposes the Self-in-Cross-Attention (SinCA) module, which replaces discrete feature matching with continuous feature transformation, and introduces low-quality feature into cross attention by the input-enhanced codebook, achieveing better quantitative and qualitative performance on both BSR and BFR tasks.

### Strengths
1. The paper conducts a large number of experiments to demonstrate the validity of the three observations of VQ on BIR task. These three observations significantly exist in the existing VQ-based methods, resulting in poor performance.

2. A self-in-cross-attention module is proposed to tackle the issue of the discrete VQ process in code index prediction by the continuous feature transformation.  The idea about using LQ image feature and HQ codebook in cross attetion is simple but effective.

### Weaknesses
Although the author's demonstration of the shortcomings of VQ discrete matching is relatively sufficient, it is not novel, and there are also some problems in some places. 

1. Originality is somewhat lacking: in Section 4.1,  the codeword utilization rates of the FeMaSR and AdaCode algorithms in the BSR task are low, and  Section 4.2 indicates that the encoding prediction accuracy of the CodeFormer and DAEFR algorithms in the BFR task is low. These two key issues have actually been described in previous works and is not the first time they have been brought up by the author. Many works have proposed solutions to the codebook collapse, and the problem of low prediction accuracy has actually been depicted in related charts in CodeFormer (Fig 6 in ref[1], the accuracy of Transformer encoding prediction is greater than that of the nearest neighbor, but still relatively low).
 
       [1] Shangchen Zhou, Kelvin Chan, Chongyi Li, and Chen Change Loy. Towards robust blind face restoration with codebook lookup transformer. Adv. Neural Inform. Process. Syst., 35:30599– 30611, 2022.

2. experiments are insufficient： Could you demonstrate that  the proposed SinCA module can ensure that the image quality does not degrade while enhancing fidelity (visualization + more comprehensive subjective + objective indicator data proof), which has no connection with the feature fusion module. In Section 4.3, by removing the feature fusion modules of FeMaSR, AdaCode, and CodeFormer, it is discovered that the PSNR, SSIM, and LPIPS indicators decline. This is used to prove the significance of the information in the input low-quality images (whether it is the multi-scale features in the Encoder or the low-quality compressed features before feature matching). A crucial point in the BFR task is actually overlooked, namely, maintaining the balance between fidelity and quality. For instance, CodeFormer achieves this through the hyperparameter w. If w is 0, then the CFT feature fusion module does not function, ensuring that no low-quality information enters, thereby enhancing the image quality compared to the situation where CFT is effective. The decrease in subjective indicators in this case is evident, but the objective indicators are likely to improve, especially in the case of severely degraded input. 

3. The contibution is not enough, because the paper mainly includes several obsevations with slightly lese novelty and propose a simple attention module SinCA.

### Questions
Could you provide a more comprehensive experiment to show that introducing low-quality features can achieve high-quality restoration results on heavy degradation restoration tasks (such as the WebPhoto/Wider datasets of human faces) while improving fidelity on more real-world datasets for BFR tasks other than CelebAChild.

### Soundness
2

### Presentation
3

### Contribution
2
