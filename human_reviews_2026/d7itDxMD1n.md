# Monocular Normal Estimation via Shading Sequence Estimation

- Avg Score: 6.40
- Decision: Accept (Oral)
- Scores: 6, 8, 6, 6, 6

## Abstract
Monocular normal estimation aims to estimate the normal map from a single RGB image of an object under arbitrary lights. Existing methods rely on deep models to directly predict normal maps. However, they often suffer from 3D misalignment: while the estimated normal maps may appear to have a correct appearance, the reconstructed surfaces often fail to align with the 3D geometry. We argue that this misalignment stems from the current paradigm: the model struggles to distinguish and estimate varying geometry represented in normal maps, as the differences in underlying geometry are reflected only through relatively subtle color variations. To address this issue, we propose a new paradigm that reformulates normal estimation as shading sequence estimation, where shading sequences are more sensitive to various geometry information. By learning to infer the shading sequence of an object, the model can better capture underlying 3D geometry and thereby produce more accurate normal predictions. Building on this paradigm, we present RoSE, a method that leverages image-to-video generative models to predict shading sequences, which are then converted into normal maps by solving a simple ordinary least-squares problem. To enhance robustness and better handle complex objects, RoSE is trained on a synthetic dataset, MultiShade, with diverse shapes, materials, and light conditions. Experiments demonstrate that RoSE achieves state-of-the-art performance on both synthetic and real-world benchmark datasets for object-based monocular normal estimation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper targets monocular normal map estimation, which aims to recover surface normal information from a single RGB object image under arbitrary lighting conditions. The core technical approach is to formulate the normal estimation problem as a shading sequence estimation task, using an image-to-video generative model to predict a sequence of shading images under predefined parallel light directions from a grayscale input image. This sequence is then converted into a normal map using an ordinary least square solver. The main contributions are to introduce the shading sequence as a more sensitive geometric representation to address 3D misalignment issues. Employing a video diffusion model to generate consistent shading sequences is also interesting. A new synthetic dataset MultiShade is also proposed. Experimental results demonstrate that RoSE achieves state-of-the-art performance on real-world benchmark datasets such as DiLiGenT and LUCES, while also excels on the synthetic MultiShade dataset.

### Strengths
The paper has some clear strengths:
Transforming normal estimation into a shading sequence generation task and leveraging video generative models to handle temporal consistency are very interesting choice. This introduces a new perspective for normal reconstruction.

The paper uses shading sequences as the training target, which is more sensitive to geometric variations than directly predicting normal maps.

The method achieves state-of-the-art performance on real-world benchmark datasets.

### Weaknesses
The key idea and technical module, which uses a video generation model to predict shading sequences, is similar to NormalCrafter [Bin et al 2025]. However, it seems that it is not discussed in this paper.

I'm not sure if using grayscale input images is a good idea. Despite being claimed in the paper that it can "eliminating redundant chromatic information", but did you conduct any ablation studies on this? For complex materials (e.g. metallic reflections), color may provide additional geometric hints, potentially leading to information loss.

The key claim of “reducing 3D misalignment” is only supported by qualitative comparisons and MAE metrics, without quantifying 3D reconstruction errors. If we use RoSE to estimate multi-view normal maps of a single object, can we achieve better consistency and reconstruction accuracy?

Estimating monocular normals using dense video models consumes excessive additional resources. At this point, I'm still uncertain whether such overhead would be widely accepted in practical applications.

The paper claims directly estimating single normal maps will lead to 3D inconsistency. I think multi-view normal estimation may alleviate this problem to some extent. The representative works such as Unique3D and Era3D also show detailed normal estimation results via a multi-view setting, but they are not compared and discussed in the paper.

### Questions
Will Negative-clamping introduce nonlinearity, which may violate the OLS linearity assumption?  It is not a perfect linear equivalent.

The choice of latitude for the ring light is described as an “appropriate choice” without specifying exact values or a computational formula? Is ring light setup the best choice? How do we "ensuring that each point is illuminated by at least three sources with positive shading values"?

Just curious: How can we verify that the improvement in normal estimation accuracy comes from estimating shading, rather than the extremely dense information provided by the video model? Is it possible to achieve similar results by estimating depth or directly estimating normals instead?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes RoSE, a monocular surface normal estimator that reformulates normal estimation as “shading sequence estimation.” Instead of directly predicting a normal map from a single RGB image, RoSE leverages an image-to-video diffusion model to generate a short sequence of grayscale, lambertian shadings of the input object under a predefined set of parallel lights; the final normals are then recovered analytically via Ordinary Least Squares (OLS). This change of target is motivated by the claim that shading sequences are more sensitive to geometric variations than the compact color encoding of normal maps, reducing “3D misalignment” (over-smooth or geometrically inconsistent normals). To train and evaluate the model, the paper also curate a synthetic dataset MultiShade with diverse shapes, materials, and lighting. Experiments on DiLiGenT, LUCES, and the proposed MultiShade dataset show state-of-the-art performance among monocular normal estimation methods.

### Strengths
- The paper proposes an interesting approach that combines the generative capabilities of video models with classic shape-from-shading priors to tackle geometric reconstruction.

- Formulating the synthesis of shading images directly as a video generation problem—rather than relying primarily on image generation as in prior work is an interesting point. This also aligns with recent findings suggesting that video generators already possess strong general-purpose visual understanding and synthesis abilities [1].

- The experimental results appear strong and promising.

### Weaknesses
- The generated Shading Sequence is central to the method, yet the experiments lack both qualitative and quantitative evaluations of this component (see Questions below).

- The discussion of output variation (L357) and the failure cases (last section of the appendix) is too generic. At L357, the inconsistencies across cases are attributed to “the inherent variance of the model,” which is uninformative and unsatisfying. More concrete analysis is encouraged: for example, examining how input illumination, texture complexity, and BRDF properties influence the recovered normal accuracy.

- The paper would benefit from additional visualizations, especially examples of the generated Shading Sequence and relighting videos.

### Questions
- The closed-form OLS solution for normal map estimation requires reasonably accurate shading sequences. How robust is the generated Shading Sequence in practice? If there is bias in the estimated shading sequence from the network, to what extent does it degrade the final normal quality?

- A classic way to incorporate shape-from-shading into deep learning is via a render-loss (dating back at least to [2]), where predicted normals/SVBRDF are rendered under multiple point lights and gradients are backpropagated. When training on synthetic datasets, it is possible to guarantee that the lighting condition used for render loss also exactly align with the proposed lighting setup in this paper. Hence an interesting question is: how do the trade-offs compare between (a) training with a render-loss, i.e., the network predicts normals directly but the loss is render (shading)-aware during training), and (b) the proposed approach in this paper (first generate the shading, then solve for normals offline)? 

[1] Wiedemer, Thaddäus, et al. “Video models are zero-shot learners and reasoners.” arXiv:2509.20328 (2025).

[2] Deschaintre, Valentin, et al. “Single-image SVBRDF capture with a rendering-aware deep network.” ACM TOG 37.4 (2018): 1–15.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the long-standing 3D misalignment issue in monocular normal estimation by introducing a novel framework, RoSE. The proposed paradigm reconceptualizes normal estimation as a shading sequence estimation problem and leverages an image-to-video generative model to predict shading sequences under predefined parallel lighting conditions. The normal map is subsequently derived through a simple ordinary least squares formulation. Extensive experiments demonstrate that RoSE achieves state-of-the-art performance on multiple real-world benchmark datasets for object-level monocular normal estimation.

### Strengths
Originality. The paper introduces a novel approach to monocular normal estimation by rethinking the problem in the context of shading sequence estimation. This conceptual shift is quite original, as traditional methods in normal estimation usually focus on individual lighting models or directly predicting normals from images. 

Quality. The paper delivers high-quality research with state-of-the-art results on multiple real-world benchmark datasets. Achieving superior performance compared to existing methods demonstrates the technical rigor of the proposed framework. The extensive experimental evaluation suggests that the approach has been thoroughly tested and validated, which is essential in establishing the robustness of the proposed method.

Clarity. The paper appears to be clear in presenting its approach. The explanation of the framework—reconceptualizing normal estimation as a shading sequence estimation problem—is concise and understandable. The use of an ordinary least squares formulation to derive the normal map is presented as a simple final step, which helps make the approach accessible without sacrificing technical detail.

Significance. The significance of this paper lies in addressing a long-standing challenge in computer vision, monocular normal estimation under varying lighting conditions. The ability to generate accurate normal maps from monocular images has numerous applications in 3D reconstruction, robotics, AR/VR, and object recognition. By introducing a novel framework that is not heavily reliant on complex sensor setups or assumptions about lighting, the proposed work could have a substantial impact on various real-world applications where obtaining ground truth normal maps is challenging.

### Weaknesses
1.Insufficient training details. While the paper states that RoSE is built upon the SV3D model (Voleti et al., 2024), the specific parameter configurations, training protocols, and dataset utilization are not described. Providing these details would significantly improve reproducibility.

2.Limited ablation studies. The current ablation experiments primarily focus on dataset influence. It would be valuable to include analyses of the model components — such as varying SV3D settings or substituting alternative video diffusion models — to better assess the robustness and generality of the approach.

### Questions
1.What is the overall inference speed of the RoSE pipeline, including shading sequence generation and normal reconstruction?

2.How does the proposed method perform when estimating normals for transparent or translucent objects, where shading cues may be ambiguous or physically inconsistent?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the RoSE method to solve the 3D misalignment challenge in Monocular Normal Estimation (MNE). The authors reframe the MNE task from direct normal map prediction to estimating a shading sequence. This sequence, which captures pixel brightness under controlled lighting, is considered more sensitive to fine geometric details, providing a superior training signal. ROSE is trained using the new MultiShade dataset to enhance robustness. Experiments show the method achieves state-of-the-art performance on real-world benchmarks, successfully reconstructing finer geometric details with better 3D alignment.

### Strengths
1. It reformulates the task as shading sequence estimation effectively addresses the core issue of 3D misalignment and oversmoothness. It is a novel new formulation.
2. The new synthetic dataset includes diverse materials, light conditions, and material augmentation, successfully improving the model's generalization ability. The new dataset is also a contribution.
3. It achieves superior performance on key real-world benchmark datasets like DiLiGenT and LUCES.

### Weaknesses
1. The use of an image-to-video diffusion model for sequence generation introduces significant computational overhead, which may limit its use in real-time or resource-constrained applications.
2. The current evaluation is restricted to object-centric normal estimation, and generalizing the approach to complex scene-centric (indoor/outdoor) settings remains a key direction for future work.

### Questions
See Weakness

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the task of estimating surface normal maps from a single RGB image. As I understand it, the motivation stems from the observation that prior single-image normal estimators often produce results that appear color-correct but fail to align with true surface geometry. The so-called “3D misalignment” problem in the paper. The idea of this paper is that, instead of directly predicting normal maps, the authors propose leveraging shading information, which, as shown in Figure 3, is more sensitive to surface detail (assuming Figure 3 is accurate). This provides both intuition and validation for their approach.

Building on this idea, the authors design a new normal estimation framework that takes advantage of a pretrained video diffusion model. Rather than training the model to predict normals directly, they fine-tune the video diffusion network to generate a sequence of shading images. These predicted shading images are then used to recover the final normal map via a solver.

Experimental results demonstrate that this method achieves state-of-the-art performance in single-view normal map estimation.

In summary, I find the motivation and design of this paper reasonable and well-aligned. I personally like the core idea, especially if there are no prior works that attempt to estimate normal maps from single RGB images through shading prediction (In the scope of neural network normal prediction from monocular images)

As for weaknesses, the paper could be improved by including more visualization materials to help readers better understand the advantages and effectiveness of the proposed approach. Details are provided below.

### Strengths
- I appreciate the motivation and core observation of this paper, as well as how the method design naturally follows from the motivation. Specifically, I like the logical flow, identifying an underlying mechanism that may cause problems, analyzing it, and addressing it through targeted design choices. The approach feels somewhat “old school,” but I personally find it appealing.

- The experimental results are good and convincing.

### Weaknesses
- The figures in the paper are of low resolution. Many images appear blurry and show visible JPEG compression artifacts when zoomed in, making it difficult to discern differences compared to the baseline methods.

- As I mentioned in the paper summary, I think the authors could include more visualization results to help readers better appreciate the framework. For example, in the prediction pipeline, the authors use a video diffusion model to predict sequences of shading images. Could the authors provide video visualizations of these sequences to show how they look?

### Questions
Although I like the main idea, I am curious whether Figure 3 is truly accurate. According to Equation 1, the shading map is a linear projection (the dot product between the normal vector and the light direction). As far as I understand, if the normal map itself has “3D misalignment”, as the paper describes, where if neighboring values are similar, then its linearly projected shading map should also show similar misalignment, since the neighboring values would remain correlated. Could the authors clarify this point further?

In fact, I suspect that the main reason why learning shading images works better is because shading images are more natural than normal maps. That is, they are more consistent with the distribution or prior learned by the pretrained diffusion model. I would appreciate if the authors could help address this hypothesis.

### Soundness
3

### Presentation
2

### Contribution
3
