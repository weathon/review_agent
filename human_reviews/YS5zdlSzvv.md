# Unleashing the Power of Deep Dehazing Models: A Physics-guided Parametric Augmentation Net for Image Rehazing

- Avg Score: 5.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 5, 5, 5

## Abstract
Image dehazing faces significant challenges in real-world scenarios due to the large domain gap between synthetic and real-world hazy images, which often hinders dehazing performance. Collecting real-world datasets is particularly difficult, as hazy and clean image pairs must be captured under identical conditions. To address this, we propose a Physics-guided Parametric Augmentation Network (PANet) that generates realistic hazy and clean training pairs, enhancing dehazing performance in real-world applications. PANet consists of two components: a Haze-to-Parameter Mapper (HPM), which projects hazy images into a parametric space representing haze characteristics, and a Parameter-to-Haze Mapper (PHM), which converts resampled haze parameters back into hazy images. By resampling individual haze parameter maps at the pixel level in the parametric space, PANet generates diverse hazy images with physically explainable haze conditions that are not present in the training data. Our experimental results show that PANet effectively enriches existing hazy image benchmarks, significantly improving the performance of current dehazing models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents a data augmentation pipeline specifically designed for real-world dehazing. This method utilizes a physical scattering model of haze, adjusting model parameters estimated by neural networks. For each real-world training patch, the approach can generate an arbitrary number of new patches with varying haze densities. The authors validate the proposed method by demonstrating its capacity to enhance the real-world dehazing performance of three state-of-the-art dehazing techniques across four datasets.

### Strengths
This work possesses several strengths:

- It effectively enhances the richness of real-world dehazing training data, thereby improving the real-world dehazing performance of existing methods.

- It allows users to create an arbitrary number of new patches with varying haze densities.

### Weaknesses
However, the work exhibits several shortcomings:

- The contributions appear insufficient. Although it shows slight improvements over [1], the core concept remains quite similar. The claimed distinctions, such as the GAN structure and global haze adjustment, can be categorized as engineering problems rather than scientific advancements. The proposed method, which employs simple ResBlocks and pixel-wise haze adjustment, may be viewed as incremental.

- The proposed method has not been compared with existing data augmentation techniques for dehazing via DeHamer and DW-GAN on the other three datasets. While such augmentation methods could enhance existing dehazing approaches, it is essential to assess the generalizability of these improvements. However, the work neglects to compare its method against these data augmentation techniques for more general cases.

- The manuscript requires revision. For instance, L237 references Eq. ??. Additionally, Figures 2 and 3 are nearly identical, differing only in minor content details.

- The work evaluates visual performance on RTTS, which is designed for haze detection. It would be beneficial to conduct experiments on dehazing in the context of object detection to assess how real-world dehazing after data augmentation impacts downstream object detection tasks.

- According to Table 5, a larger number of augmented data pairs improves dehazing performance, yet the authors stop at 600. It would be useful to evaluate the convergence of dehazing performance relative to the number of augmentations. 

[1] Self-augmented unpaired image dehazing via density and depth decomposition, CVPR 2022.  
[2] RIDCP: Revitalizing real image dehazing via high-quality codebook priors, CVPR 2023.

### Questions
The questions have been outlined in the section on weaknesses. Considering the aforementioned strengths and weaknesses, I would recommend a borderline reject, with the potential for reconsideration if the listed issues are adequately addressed.

### Soundness
2

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
5

### Summary
This paper introduces a Physics-guided Parametric Augmentation Network (PANet) to address the domain gaps between synthetic and real-world haze data. PANet is designed to generate real haze images along with their corresponding clean pairs. It consists of two components: the Haze-to-Parameter Mapper (HPM), which projects hazy images into a parametric space, and the Parameter-to-Haze Mapper (PHM), which maps haze parameters back to hazy images.

### Strengths
1. The paper tries to address a meaningful problem: bridging domain gaps between synthetic and real-world data. 
2. The structure and presentation of the paper are clear and well-organized.

### Weaknesses
1. The authors do not account for the idealized assumptions of the physical scattering model, which may lead to inaccuracies in haze removal.
2. The importance of using real-world natural haze images, beyond the non-homogeneous haze created by fog machines, is overlooked.
3. The proposed approach heavily relies on existing datasets, which lack diversity (environment, light condition, etc.).

### Questions
1. The physical scattering model is an idealized approximation and may not accurately represent real-world haze, which can still cause domain gaps. Have the authors considered this limitation, and are there any solutions to mitigate it?
2. The training dataset NH-Haze20 is generated using a fog machine, which creates domain discrepancies between these images and those with natural real-world haze. Even with the proposed augmentation method, this gap may persist. Do the authors have any explanations or proposed solutions for this issue?
3. In Figures 6 and 7, all qualitative results appear to be zoomed-in versions. Could the authors provide full images, particularly for real-world natural images with dense haze? These images should represent natural haze rather than artificial fog.
4. The augmented dataset is based on NH-Haze20 and similar real-world datasets, which are limited and lack diversity (e.g., environmental and lighting conditions). How do the authors plan to address the reliance on these existing datasets?

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
4

### Summary
Collecting real-world hazy-clean image pairs is particularly difficult and the authors tried to address this issue by proposing PANet. PANet can generate realistic hazy and clean training pairs, thus enhancing dehazing performance in real-world applications.

### Strengths
1. the key idea of performing parametric augmentation to generate additional haze patterns is good.
2. the experimental results are promising.
3. the paper is well-prepared and easy to follow.

### Weaknesses
1. the depth refinement module (DRM) is employed to refine the initial depth map, which means the depth estimation is not accurate enough in some cases. Have the authors attempted to utilize other methods of depth estimation which are more accurate?
2. the choice of baseline method lacks convincingness. The three baseline models, DW-GAN, Dehamer, and FocalNet are primarily utilized for synthetic data (i.e., SOTS-indoor, SOTS-outdoor). Can this method be applied to real-world dehazing models (e.g., RIDCP DAD)?
3. Comparisons with methods such as RIDCP DAD PTTD, which are oriented towards real image dehazing, are lacking. In addition, by observing the images, the qualitative results in Figure 7 contain some artifacts.
4. the parametric augmentation of haze is not flexible, can the value of $\alpha$ be continuous? What's the range of values for $\alpha$? What parameters were used in the experiments section?
5. the experiments section is not sufficiently comprehensive. For real-world hazy environments, only RTTS is tested and only NIQE and PIQE are adopted as the metrics.
6. some typos: e.g., Ln237.

### Questions
Please check the weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper introduces the Physics-guided Parametric Augmentation Network (PANet), designed to improve real-world image dehazing. 

PANet combines physics-based modeling with data-driven techniques to generate diverse hazy images, aiming to bridge the gap between synthetic and real-world hazy datasets. 

By mapping haze characteristics into a parametric space, PANet can resample parameters and generate new, physically realistic hazy images.

### Strengths
- physics-guided + data-driven make sense. 

- The physics-guided and parametric approach to generating realistic hazy images also makes sense.

### Weaknesses
- The progress of daytime dehazing or defogging has been significant over the past 10 years. These methods can handle many problems, particularly when the haze or fog is relatively thin. Non-uniform haze/fog is also not a significant issue, as many methods can handle it well. (If there is any disagreement, the paper should provide evidence of existing methods failing to deal with non-uniform haze.) The main challenge of dehazing arises when the haze/fog is significantly thick. Unfortunately, the proposed method does not address this thick haze/fog problem specifically, as evidenced by the results. Moreover, the proposed method has no specific mechanism or treatment in dealing with the thick haze/fog and its characteristics.

- The qualitative experimental results do not show that the proposed method outperforms the existing methods. 
In Fig. 1 and 6, when the fog/haze is thick, the method still suffers from it and suffers from colour shift.

- The proposed method does not have any specific features that differentiate it from existing methods in terms of the haze/fog problem it aims to solve. The results presented in the paper could be achieved by existing methods, including non-deep learning methods, with comparable quality.

- Missing citation and dataset in [1]
[1] Structure Representation Network and Uncertainty Feedback Learning for Dense Non-Uniform Fog Removal

### Questions
1. For the colour shift issue, what is the reason?

2. It seems for the sky, white object and road, the results are not promising, what is the reason?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses the real data-scarcity issue for dehazing: that real-world haze is often dense and non-homogeneous, which is difficult to synthesize using traditional image formation models. The proposed haze data augmentation technique (PANet) adopts a hybrid approach, combining the strengths of both data-driven or physics-based methods. It first estimates haze parameters from clean-hazy image pairs using Haze to Parameter Mapper. In the Parameter to Haze mapper: it leverages physics-guided scattering model to generate initial hazy images. It further incorporates a Data-driven Haze Refiner (DHR) to refine this initial hazy images to enable better realism and accuracy.

### Strengths
The paper addresses a practical problem in dehazing: real-world haze is often dense and non-homogeneous, which is difficult to synthesize purely using physical scattering image formation models.

The HPM+PHM cyclic approach for unsupervised learning of intermediate haze parameters is practically effective.

Applying the proposed augmentation on selected Dehazing methods leads to notable improvement in dehazing quality on real images and few synthetic test images.

The approach is data efficient, in which it can be trained on a small dataset of as few as 50 images. The hybrid formulation leads to fewer unwanted artifacts than GAN based augmentation approaches.

### Weaknesses
Limited technical novelty: The approach is derived from existing, established methods for cyclic image-to-image mapping, specifically built upon CycleGAN.

Dataset limitations: The analysis and evidence for validating the idea are limited, as the validation relies on a small real-world dataset (NH-Haze20) for training, with only 50 training pairs and 5 testing pairs. This limited dataset size may restrict the generalizability and effectiveness of PANet in handling the diversity of real-world haze conditions.

Computational footprint and scalability: PANet is a relatively complex architecture with multiple components, including encoders, decoders, a depth refinement module, and a data-driven haze refiner. This complexity requires significant FLOPs and increases the computational cost and training time compared to simpler augmentation techniques. Additionally, how well PANet scales to larger datasets (on the order of 10^4 to 10^6 images) should be discussed.

Few writing quality issues: There are some quality issues in writing, such as an equation reference error on line 238 and typos like “pixel-wisely” on line 307.

Outdoor vs. indoor image improvement: The improvement on outdoor hazy images appears to be higher than on indoor hazy images. This observation should be discussed further.

Qualitative results clarity: It is not clear which of the three dehazing models was used to generate qualitative results, such as those in Figs. 6 and 7.

Choice of augmentations: Some choices of augmentation, such as “reverse its haze location,” seem less realistic, as they are opposite to the general nature of haze (which typically increases with distance). It would be interesting to analyze the effect of excluding such augmentations.

Dependency on DHR: The results in Table 3 suggest that the entire approach fails if the Depth Haze Refiner (DHR) is not included, which is surprising and questions the method’s utility. There should be an analysis with quantitative and qualitative results on the effect of the Depth Estimator and DRM on the performance. Additionally, extensive visualizations showing the outputs of the depth estimator, DRM, beta(z), and final t(z) are recommended.

Reliance on pre-trained depth estimator: PANet relies on a pre-trained depth estimator (RA-Depth) to estimate depth maps from clean images, which may pose a potential weakness. This estimator may not generalize well to unseen images, especially those with characteristics different from its training data. This generalization issue may not always be addressable by training a DRM, which could lead to inaccurate depth estimations and negatively impact the accuracy of the physical scattering model used in PANet, affecting the realism of the generated hazy images.

Baseline model performance: The results of three baseline dehazing models on real images from the RTTS dataset appear to be quite poor, with significant artifacts. It would be interesting to know whether any existing dehazing model can yield reasonable results on the RTTS dataset.

Selection of dehazing models: How were the three dehazing models selected? Additionally, it might be interesting to analyze any improvements observed when using other recent dehazing models.

Risk of overfitting: The potential for overfitting needs to be carefully considered. While PANet shows improvements in dehazing performance on a few similar datasets and one additional real dataset, the use of augmented data can increase the risk of overfitting, especially with a limited original dataset.

Additional metrics: Including additional no-reference metrics, such as FADE, BRISQUE, NIMA, and US, for the RTTS datasets would enable a fuller comparison with RIDCP (Wu et al., 2023).

Evaluation on popular benchmarks: The proposed approach could also be evaluated on popular dehazing benchmarks like the SOTS-Outdoor and SOTS-Indoor datasets.

Evaluation under challenging conditions: Optionally, it might be interesting to test PANet under extremely challenging haze conditions, such as dense fog or heavy smog.

### Questions
Please address weaknesses above.

### Soundness
2

### Presentation
3

### Contribution
2
