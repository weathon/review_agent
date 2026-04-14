# Learning Color Equivariant Representations

- Decision: Accept (Poster)
- Scores: 3, 5, 6, 8, 6

## Abstract
In this paper, we introduce group convolutional neural networks (GCNNs) equivariant to color variation. GCNNs have been designed for a variety of geometric transformations from 2D and 3D rotation groups, to semi-groups such as scale. Despite the improved interpretability, accuracy and generalizability of these architectures, GCNNs have seen limited application in the context of perceptual quantities. Notably, the recent CEConv network uses a GCNN to achieve equivariance to hue transformations by convolving input images with a hue rotated RGB filter. However, this approach leads to invalid RGB values which break equivariance and degrade performance. We resolve these issues with a lifting layer that transforms the input image directly, thereby circumventing the issue of invalid RGB values and improving equivariance error by over three orders of magnitude. Moreover, we extend the notion of color equivariance to include equivariance to saturation and luminance shift. Our hue-, saturation-, luminance- and color-equivariant networks achieve strong generalization to out-of-distribution perceptual variations and improved sample efficiency over conventional architectures. We demonstrate the utility of our approach on synthetic and real world datasets where we consistently outperform competitive baselines.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The authors propose GCNNs which are equivariant to color variation. GCNNs have been widely used for geometric transformations while only a few research has attempted to apply GCNNs to color-equivariant networks to achieve generalization to out-of-distribution samples. Throughout the paper, the authors reasonably justified the reason for designing the color-equivariant networks and proposed a proper method for dealing with such situations. Both quantitative and qualitative evaluations are conducted in the paper.

### Strengths
- Interpretable visual illustrations of the results
- Interesting motivation
- Decent paper writing

### Weaknesses
Overall, it is clear to me that the proposed method is better than the baselines (ResNet50, Z2CNN) under the generalization testing setup. However, the improvements over CEConvs look marginal (performance-wise and method-wise). Other weaknesses I have found are:

1. Related Works: 
    - What is the connection/difference of the proposed setting with Domain Generalization?
2. Missing baselines to support the argument in L44-50.
    - converting input image to grayscale 
    - data augmentation

### Questions
1. Fig. 1: Baseline comparison with CEConvs

2. Fig. 4
    - What is the “Model sample efficiency”? 
    - How can the error improvement be computed?
    - Fig. 4 shows that there is no improvement over Z2CNN if 10% or 100% of training example is used. Why does it happen?
    - On the other hand, the advantage of the proposed methods are shown effective when only a little portion of the training example is used. What does this mean?

3. Table 1: Baseline comparison with CEConvs

4. Fig 5: Baseline comparison with CEConvs

5. Fig 7: Baseline comparison with CEConvs

6. What is the difference between the global hue-shift out-of-distribution (A/B) and the local hue-shift out-of-distribution (A/C)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper builds on the approach “Color Equivariant Convolutional Networks” [16, Neurips 2023], modifying their lifting layer by operating on the input image rather than on the filters. Additionally, it generalizes the equivariance, previously only defined on the hue channel to the  saturation channel. Experiments on synthetic datasets (hue shift MNIST and hue shift 3D shape) and on CIFAR, Caltech, Camelyon17, Caltech, STL-101, Stanford Cars, and Oxford pets dataset show improved classification performance compared to the CE-conv3 approach [16] and to Resnet baselines with same number of parameters.

### Strengths
* The paper is well written and illustrated.
* The approach is well motivated and addresses concrete tasks, for instance improving classification robustness in case of different imaging processes coming from different labs in the case of medical imagery. 
* The claims are supported by experiments. 
* Some limitations of the approach are presented.  
* The code will be open source.
* The appendix provides necessary information to reproduce results.
* The approach designs color equivariance in group CCNN models, which is an interesting property.

### Weaknesses
* The claim of improved equivariance by order of magnitude over [16] on the 3D shape dataset is true but the error of [16] is already very low on this dataset, 0.05 at most for CEconv-3. As the error is very small, this difference in magnitude will not translate into drastic improvements in practice.
* [Comparison to stronger baselines] The comparison to [16] on non-synthetic datasets displays larger errors than in [16]. Why not use their experimental protocol that seems well detailed in their paper? 
[Incomplete comparison] Only results with CE-conv3 and CE-conv4 appear in the current paper, while [16] often shows better results with CE-conv2.
* [No comparison to SOTA] Further discussion on the Camelyon-17 results would be interesting: Looking at the leaderboard on this dataset, https://wilds.stanford.edu/leaderboard/ , we note a number of approaches are performing better than the proposed approach. What would be the pros and cons of using the proposed approach vs the SOTA ?       
* The cardinality N of the group used in the experiments is not discussed. In addition, it is only introduced on page 7, it would help the reader to define the notation before as all figures are using it.

### Questions
1/ Why not use the experimental protocol of [16] that seems well detailed in their paper for Table 4?

2/ What would be the pros and cons of using the proposed approach vs the SOTA (or a strong baseline from the leaderboard such as Group DRO) on Camelyon17 ?

3/ Could you explain how the different cardinalities have been chosen for the different experiments?

4/ In the limitation section, the computation overhead of the approach is mentioned, without mentioning explicit numbers compared to the baseline’s training time, could this information be mentioned?

### Soundness
4

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
3

### Summary
The paper presents a novel convolutional neural network (CNN) architecture tailored to handle color variations effectively. The authors introduce a color-equivariant group convolutional network (GCNN) that is robust to changes in hue and saturation, significantly improving performance in tasks involving perceptual color transformations. By employing a new lifting layer that operates directly on input images rather than filters, the model overcomes limitations of existing color-equivariant networks, achieving enhanced stability and reducing equivariance errors by orders of magnitude. This approach yields superior performance on both synthetic and real-world datasets, particularly excelling in generalizing to out-of-distribution color variations and enhancing sample efficiency.

### Strengths
1, The paper presents a novel approach to color-equivariant convolutional neural networks (CNNs) by extending GCNNs to handle hue and saturation changes in the HSL color space. This is achieved with an inventive lifting layer that acts directly on input images, improving stability and reducing errors by avoiding invalid RGB values—a common issue in previous architectures.

2, The authors demonstrate a thorough experimental design using various synthetic and real-world datasets (e.g., Hue-shift MNIST, Camelyon17) known for color variations, showing the model’s robustness. Both quantitative and qualitative analyses, including sample efficiency and feature map visualizations, offer a detailed evaluation. Comparisons with competitive baselines like CEConv and ResNet confirm the model’s generalizability and efficiency.

3, The paper is well-structured, with a logical flow from problem statement to methodology and results. Clear explanations of the hue- and saturation-equivariant group actions, along with detailed figures, aid in understanding the technical concepts. Comprehensive appendices further enhance reproducibility and transparency by providing proofs and additional experiment details.

### Weaknesses
1, The paper extends GCNNs to be equivariant to hue and saturation but does not address luminance variations, which are also important for color representation. Although the authors suggest that luminance invariance may be approximately preserved, this may not hold in real-world applications, such as medical imaging or scenes with varying lighting. Future work could integrate luminance into equivariant transformations or evaluate its impact on specific tasks.

2, The model is compared to standard CNNs (e.g., ResNet) and color-equivariant networks (e.g., CEConv), but not to other common approaches for handling color variations, such as color augmentation techniques or robust contrastive learning models. Adding these comparisons would strengthen the case for this method’s effectiveness in handling color shifts.

### Questions
1, The additional computational load from group convolutions and lifting layers present a challenge for real-time applications. Therefore, it is essential to compare your method with common alternatives, such as color augmentation or robust contrastive learning approaches, which are known to enhance robustness to color and perceptual variations. These comparisons would be especially valuable for demonstrating the unique advantages of color equivariance over standard augmentation techniques.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper extends the concept of color equivariance to include saturation shifts, in addition to hue transformations, then they design GCNNs that are equivariant to color variation by leveraging the geometric structure of color in the hue-saturation-luminance (HSL) color space, additionally a novel lifting layer is proposed to transform input images directly, avoiding invalid RGB values and significantly improving equivariance error.

### Strengths
1. The paper is very well written and easy to read.

2. The authors conducted extensive experiments using natural images (CIFAR10, CIFAR100, STL-10, etc.) as well as medical images (Camelyon-17) to validate the effectiveness of their algorithm. The experiments are thorough and detailed.

3. Color equivariant representations are an important concept in computer vision, yet they are often overlooked in contemporary research. The authors effectively claim the significance of color in the era of deep learning in this work.

### Weaknesses
I believe this work has no significant drawbacks and meets the acceptance standards for ICLR. 

Beyond that, I am curious about two questions:

1. This paper uses the HSL color space as the basis for designing experiments, specifically designing groups and group actions for hue and saturation, as well as a hue-saturation group action. I am interested to know whether this approach would also be applicable to other chromacity-luminance color spaces, such as CIELAB or YUV.  Additionally, the experiments are designed around hue shifts in the HSI series (i.e. H in HSV. HSL, HSB...). Could other chromacity shifts in different color spaces be designed to further demonstrate the robustness of the model?

2. The experiments presented in this paper focus on classification tasks. I wonder if this design could be even more beneficial for higher-level tasks or those that rely more on semantic information, such as semantic segmentation. Do the authors have any plans to design similar experiments for such tasks?

### Questions
See weakness part.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces group convolutional neural networks designed to be equivariant to variations in hue, saturation, and color in RGB images. It proposes a lifting layer that directly transforms the input image to produce a color-equivariant descriptor, effectively addressing the challenge of invalid RGB values found in previous works. Experiments on synthetic and real-world datasets highlight the robustness of the proposed method to color variations.

### Strengths
1.	The approach is technically compelling, as it enables convolutional neural networks to be equivariant to hue and saturation by design, leveraging the inherent geometric structure of these color components.

2.	Results across several benchmarks indicate that the proposed method offers stronger robustness to color variation compared to existing methods.

3.	The paper is well-structured and easy to understand.

### Weaknesses
1.	Experiments are limited to small toy datasets with pronounced hue or saturation shifts, leaving it unclear how the method would perform on large datasets like ImageNet.

2.	The proposed method focuses on hue and saturation equivariance, but does not address other common variations in real-world data, such as 3D geometric transformations or lighting changes.

### Questions
Would the proposed method improve image classification performance on large-scale datasets like ImageNet or on tasks beyond image classification?

### Soundness
2

### Presentation
3

### Contribution
2
