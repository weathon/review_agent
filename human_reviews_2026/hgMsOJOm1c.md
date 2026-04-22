# CHROMA: Consistent Harmonization of Multi-View Appearance via Bilateral Grid Prediction

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Modern camera pipelines apply extensive on-device processing, such as exposure adjustment, white balance, and color correction, which, while beneficial individually, often introduce photometric inconsistencies across views. These appearance variations violate multi-view consistency and degrade novel view synthesis. 
Joint optimization of scene-specific representations and per-image appearance embeddings has been proposed to address this issue, but with increased computational complexity and slower training. 
In this work, we propose a generalizable, feed-forward approach that predicts spatially adaptive bilateral grids to correct photometric variations in a multi-view consistent manner. Our model processes hundreds of frames in a single step, enabling efficient large-scale harmonization, and seamlessly integrates into downstream 3D reconstruction models, providing cross-scene generalization without requiring scene-specific retraining. To overcome the lack of paired data, we employ a hybrid self-supervised rendering loss leveraging 3D foundation models, improving generalization to real-world variations.
Extensive experiments show that our approach outperforms or matches the reconstruction quality of existing scene-specific optimization methods with appearance modeling, without significantly affecting the training time of baseline 3D models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper propose CHROMA, to handling novel view synthesis under lightness and ISP variances. Unlike previous methods (such as BiLaRF, Luminance-GS, Wild-Gaussian, etc.), which primarily focus on per-scene optimization and often requiring adjustments to per-scene curves, grids, or other attributes. The proposed method adopts a different solution. it trains a network by synthesizing multi-view data with various ISP degradations. This network includes a transformer encoder, decoder, and grid prediction head. Once trained, the network can serve as an effective multi-view pre-processing tool, restoring multi-view images to standard conditions. This enables the training of a range of multi-view reconstruction methods, such as 3DGS, 2DGS, DashGS, and others. Through the comparison, CHROMA achieve the SOTA results on 3 different datasets.

### Strengths
1. This work holds significant practical value. Designing a network capable of harmonizing 3D multi-view images is particularly important, especially given the widespread use of smartphone photography and handheld devices, where camera ISP variance and lightness variance are highly prevalent phenomena.

2. This work eliminates the need for per-scene adjustments on each set of multi-view images, significantly reducing complexity. Although it does not constitute a fully end-to-end network (such as Depth-Splat or MVSGaussian), it already demonstrates substantial practical value. Moreover, experimental results show that CHROMA achieves top-tier performance across all benchmark settings.

3. The paper features an intelligently designed network architecture that effectively reduces inference time and memory consumption (as demonstrated in Table 6 and Table 7), while the writing remains exceptionally fluid throughout.

### Weaknesses
1. In summary, this work represents a solid incremental contribution by applying a proven approach, as opposed to formulating a new research problem. Thus I consider its maturity and novelty to be appropriate for a poster, but not sufficient to meet the higher bar for an oral presentation.

2. Regarding the results labeled "+ours" in Table 1, does the reported time include the duration required for the CHROMA network to process the multi-view images? Overall, the reported speed of the proposed method seems almost implausible—for instance, it adds only 0.04s on the BiLaRF dataset. This could easily mislead readers. If such efficiency is indeed achievable, the authors should provide a clear explanation. Additionally, it is important to note whether the inference speed of CHROMA has been evaluated across different GPU hardware.

3. Furthermore, compared to methods like Luminance-GS and GS-W, the authors' approach requires training a network from scratch, which should be explicitly mentioned in the Limitations section.

### Questions
While Tables 6 and 7 effectively demonstrate the efficiency gains of the proposed components, it would be equally important to present a comparative analysis of their reconstruction performance. Could the authors provide quantitative results (e.g., PSNR, SSIM, LPIPS) corresponding to the different head designs to offer a more complete picture of the performance-efficiency trade-off?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel method for multi-view image harmonization in a feed-forward manner. By employing a VGG-style network architecture to predict a pre-frame bilateral grid, the proposed approach achieves harmonized multi-view images with superior reconstruction quality compared to existing methods.

### Strengths
1. The proposed feed-forward framework is novel and well-motivated, effectively improving illumination and color consistency across multiple views and achieving state-of-the-art performance compared to existing methods.
2. This paper is well-written and easy to read.

### Weaknesses
1.	The proposed method relies on the assumption that at least one reference view is reliable. However, in real-world scenarios where all input views contain significant artifacts or inconsistencies, the effectiveness of the proposed framework could be severely degraded.
2.	The method models inter-view inconsistencies using a patch-level bilateral grid, which may lead to visible block artifacts. Although a total variation (TV) loss is applied to alleviate this issue, the results are shown only on low-resolution inputs. It remains unclear whether the approach can maintain similar harmonization quality on high-resolution images, where block boundaries could become more noticeable.
3.	The proposed 3D Foundation Model–based self-supervised loss pipeline can improve performance to some extent, but it heavily depends on the chosen pretrained model and introduces considerable training overhead. Moreover, according to the ablation results, the performance gain from this module is marginal, and it even shows a slight decrease in SSIM, suggesting that the inclusion of this component may not be well justified.

### Questions
1. How would the method perform if all input views contain artifacts or illumination inconsistencies? Is there any mechanism or potential extension that could improve robustness in such cases?
2. Could the authors provide qualitative or quantitative results on higher-resolution inputs? Since the current experiments are mainly conducted on low-resolution images, it would be helpful to understand how the proposed method performs when applied to higher-resolution data.
3. Could the authors provide more details about the training data scale? For example, how many multi-view scenes or image pairs were used for training? 
4. How sensitive is the method to the choice of the pretrained 3D foundation model? If a different foundation model were used, how would it affect the overall performance and stability of the system?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
1. This paper studies the task of multi-view appearance harmonization. The goal is to make images of the same scene from different viewpoints consistent in color, exposure, and white balance, which helps improve 3D reconstruction and novel view synthesis quality.

2. Previous works did not focus on feed-forward harmonization. This paper is the first to explore a feed-forward solution for this task.

3. The authors propose a general transformer that predicts a 3D bilateral grid for each view to achieve consistent color mapping. The bilateral grid provides spatial adaptivity and edge preservation, making it effective for modeling ISP-related variations.

4. To train the model, the authors use DL3DV with synthetic ISP augmentations and introduce a rendering consistency loss based on a 3D reconstruction model.

### Strengths
1. The method is technically sound. The bilateral grid is well suited for spatially adaptive and edge-preserving color correction.

2. The experiments are thorough and well validated.

### Weaknesses
1. The authors train the model on DL3DV with handcrafted ISP variations and also test on DL3DV with the same synthetic setup. This limits the significance of the results, since a feed-forward model will naturally perform well under similar handcrafted conditions.

2. In Table 1, the improvement over Luminance-GS on the LOM and BilaRF datasets appears small.

### Questions
Please address the issues listed in the weaknesses section.

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
The paper presents a feed-forward approach for appearance adjustments in order to achieve photometric consistency across multiple views that can then improve the accuracy of downstream 3D reconstruction tasks. The proposed method learns spatially varying bilateral grids for edge-aware photometric adjustments. The paper additionally includes a methodology for selecting an appropriate reference frame for photometric alignment, and a self-supervised training paradigm by optimizing over the downstream 3D reconstruction achieved from consistent novel views.

### Strengths
The paper is well written and theoretically detailed. The overall evaluation presented in the paper is mostly thorough and shows favourable results from the proposed approach. It is particularly interesting to see considerable improvement in 3D reconstruction (Fig 4).

The theoretical contributions in the paper are more or less incremental when viewed in the context of existing literature, however, they seem to be put together in a coherent manner in the proposed approach. Bilateral grids for per-image appearance editing while enforcing consistency using self and cross attention are the main contributions in my opinion.

Selecting the correct reference frame is quintessential for the success of this approach, and it is good to note that the authors have discussed this at length, considering both extrema of high and low luminosity and avoiding them.

Evaluating the method across three datasets covering synthetic, exposure-stacking and wild scenes is commendable and the proposed method shows favorable results.

### Weaknesses
One of the main challenges in photometric consistency across multiple views is in being able to handle specularities - which is where most methods fail remarkably. While specular surfaces may be significantly challenging to address, varying specularities across views from bright light sources and low dynamic range of sensors is certainly applicable to the proposed approach. I would expect the approach to at least discuss and/or present the performance on some examples.

The image in column 3 of Figure 10 is particularly confusing. It seems as if a uniform scaling down of the luminance is applied here without any contrast stretching in the input image. However, all the demonstrated methods show varying levels of contrast stretching, including the proposed approach. I presume there is a mismatch between input and generated images, in my opinion.

There are some examples where the qualitative improvement is not immediately apparent, if at all, in the figures presented in Figure 11.

Figure 4 is where the true impact on geometry is visible on using the proposed approach. If one were to compare 3DGS versus 3DGS+Ours for the Bicycle scene, the fine details in the bicycle and the bench are clearly visible in the proposed method, however, it is not immediately apparent why this may be achieved by photometric consistency across views alone, particularly because of the lack of significant photometric variation in the direction of view for these parts of the scene.

As is the case with several self-supervised approaches, there is an inherent assumption here that the output of the 3D foundation model (Anysplat) is photometrically reliable. Anysplat itself relies on scene visibility, textural details and implicit sparse depth across views. Some analysis on the self-reinforcement caused by this assumption will be useful to the reader.

The confidence aware prediction mechanism is appreciated, however, I believe it is also important to illustrate some of the per-pixel confidence maps (l.242) of a few views from the same scene to actually decipher the impact of the proposed method. There is no mention of this in the paper.

Given the efficiency and easy-integration proposed in the paper, more discussion on temporal consistency may be useful for dynamic scenes and may result in a wider contribution (extending the discussion about Table 2)

Affine warping in a patch based manner (as proposed in the paper) brings with it the likelihood of washing out legitimate scene properties such as shadows - this may be useful from the standpoint of geometric reconstruction, but not so much in terms of photo-realistic view synthesis. I would expect the authors to mention or discuss this point, even if briefly.

2D-only methods like UEC and MSEC are more relevant for comparison in terms of compute time. I would urge the authors to provide that comparison.

### Questions
See points under the Weakness section

### Soundness
3

### Presentation
3

### Contribution
3
