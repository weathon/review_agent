# Sensitivity-Adaptive Augmentation for Robust Segmentation

- Decision: Reject
- Scores: 3, 6, 5, 5

## Abstract
Achieving robustness in image segmentation models is challenging due to the fine-grained nature of pixel-level classification. These models, which are crucial for many real-time perception applications, particularly struggle when faced with natural corruptions. While sensitivity analysis can help us understand how input variables influence model outputs, applying it to natural and uncontrollable corruptions in training data is difficult. In this work, we present an efficient, sensitivity-based augmentation method to enhance robustness against natural corruptions. Our sensitivity analysis approach runs up to 10 times faster and requires up to 200 times less storage than previous approaches, enabling practical, on-the-fly estimation during training for a model-free augmentation policy.
With minimal fine-tuning, our sensitivity-based augmentation method achieves improved robustness on both real-world and synthetic datasets compared to state-of-the-art data augmentation techniques in image segmentation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper introduces a technique to identify optimal data augmentation parameters that enhance the robustness of a semantic segmentation DNN. The authors focus on augmentations parameterized by a variable, alpha, examining the model's sensitivity to this parameter. To assess sensitivity, they use the Kernel Inception Distance (KID), which varies based on the sample and alpha. Additionally, they define the Metric of Accuracy (MA), which measures performance after image modification, dependent on X and alpha. The objective is to maintain stability in MA while allowing KID to vary. Sensitivity, defined as the ratio of these metrics, is minimized in theory, although, due to computational constraints, they approximate this minimization. Further, to reduce costs, the exploration of alpha is restricted to a truncated parameter space rather than the entire range.

### Strengths
The paper addresses a relevant and compelling task of identifying optimal augmentation parameters, and the aim of increasing DNN robustness is beneficial for the field.

### Weaknesses
**1. Writing Quality:**

The paper’s writing is difficult to follow and would benefit from thorough revision. I recommend that the authors dedicate more time to clarifying and refining the text.

**2. Missing Comparisons:**

The paper lacks comparisons with relevant methodologies in the field. While the authors include AutoAugment, it is not specifically trained on semantic segmentation, which limits the relevance of the comparison. It would be valuable to benchmark the author’s approach against meta-learning algorithms such as Hyperopt [1], Spearmint [2], Tree-Structured Parzen Estimator (TPE) [6], SMAC [3], Autotune [4], and Vizier [5].
Additionally, comparisons with robust semantic segmentation methods like those in [7, 8, 9, 10] would strengthen the evaluation and provide context for the proposed technique

**3. Mathematical Clarity and Accuracy:**

3.1 Further explanation of the MMD (Maximum Mean Discrepancy) and the choice of kernel, including details on the polynomial kernel, would be helpful.
3.2 Clarification is needed on the transition from Equation 3 to Equation 6.
3.3 Equation 5 appears to have an inconsistency; the denominator linked to KID differs from that in MA—could this be an error?
3.4 The KID numerator in Equation 7 is unusual; please clarify its derivation.
3.5 The definitions of alpha' and alpha'' in Equations 4 and 5 are unclear and require explanation.
3.6 The equation between Equations 8 and 9 seems incorrect. My calculations indicate a result of infinity rather than 2; could the authors verify this?
3.7 Please define T in Equation 10.
3.8 The section from lines 290 to 303 is challenging to interpret even after multiple readings. A rewrite of this section would be beneficial.

**4. Methodology Clarity:**

The paper could improve clarity regarding its approach. The claim of finding data augmentations efficiently is weakened by unclear and potentially biased comparisons. It appears unfair to compare a general approach to an optimized dataset-specific one, and there is also ambiguity in the data augmentations used by each method. Including a table detailing the data augmentations for all algorithms would help in transparency.

**5. Sensitivity Analysis:**

An analysis of the sensitivity of the proposed method to different data augmentation parameters would provide further insight and strengthen the evaluation.


[1] James Bergstra, Brent Komer, Chris Eliasmith, Dan Yamins, and David D Cox. Hyperopt: a Python library for model selection and hyperparameter optimization. Computational Science & Discovery, 8(1):14008, 2015. 

[2] Jasper Snoek, Hugo Larochelle, and Ryan P Adams. Practical bayesian optimization of machine learning algorithms. In NIPS, pages 2951–2959, 2012. 

[3] Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown. Sequential model-based optimization for general algorithm configuration. In LION, pages 507–523, 2011. ISBN 978- 3-642-25565-6.

 [4] Patrick Koch, Oleg Golovidov, Steven Gardner, Brett Wujek, Joshua Griffin, and Yan Xu. Autotune: A derivativefree optimization framework for hyperparameter tuning. In KDD, pages 443–452, 2018. ISBN 978-1-4503-5552-0.

 [5] Daniel Golovin, Benjamin Solnik, Subhodeep Moitra, Greg Kochanski, John Karro, and D Sculley. Google Vizier: A service for black-box optimization. In KDD, pages 1487–1495, 2017. ISBN 978-1-4503-4887-4.

 [6] James Bergstra, Remi Bardenet, Yoshua Bengio, and  Balazs Kegl. Algorithms for hyper-parameter optimization.  In NIPS, pages 2546–2554, 2011.

[7] Larsson, M., Stenborg, E., Hammarstrand, L., Pollefeys, M., Sattler, T., & Kahl, F. (2019). A cross-season correspondence dataset for robust semantic segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9532-9542).

[8] Xu, X., Zhao, H., & Jia, J. (2021). Dynamic divide-and-conquer adversarial training for robust semantic segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 7486-7495).

[9] Franchi, G., Belkhir, N., Ha, M. L., Hu, Y., Bursuc, A., Blanz, V., & Yao, A. (2021). Robust semantic segmentation with superpixel-mix. arXiv preprint arXiv:2108.00968.

[10] Myronenko, Andriy, and Ali Hatamizadeh. "Robust semantic segmentation of brain tumor regions from 3D MRIs." Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries: 5th International Workshop, BrainLes 2019, Held in Conjunction with MICCAI 2019, Shenzhen, China, October 17, 2019, Revised Selected Papers, Part II 5. Springer International Publishing, 2020.

### Questions
See most of the questions on the weakness.

Also please add a pseudo code describing your technique

### Soundness
2

### Presentation
2

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
The paper presents a approach for improving the robustness of segmentation models through sensitivity-based data augmentation. It attempts to address the challenges of natural corruptions in image segmentation by combining sensitivity analysis with data augmentation. The use of Kernel Inception Distance (KID) to measure the extent of augmentation and the formulation of the sensitivity analysis objective are well-motivated and provide a new perspective on improving model robustness. Expeirmental results confirm that the proposed augmentation method improves model's robustness in real-world and synthetic datasets.

### Strengths
1. The idea of using sensitivity analysis to adaptively augment sample is interesting and has the potential to enhance the performance of segmentation models in the presence of natural corruptions. 

2. The authors conduct a series of experiments on multiple datasets and in different scenarios, including real-world corruptions (ACDC and IDD datasets), various synthetic benchmarks (ADE20K, VOC2012, etc.), and fine-tuning with foundation models in downstream tasks. The comparison with several state-of-the-art augmentation methods helps to demonstrate the effectiveness of the proposed method. 

3. The paper acknowledges the limitations of the current work, such as the inability to handle physical occlusions and the relatively poor performance in low-lighting scenarios. This shows the authors' awareness of the potential areas for improvement and provides a direction for future research.

### Weaknesses
1. The description of sensitivity analysis, especially the calculation of the sensitivity curve and the interpretation of its derivative, could be more intuitive. The equations and concepts related to sensitivity analysis may be difficult for some readers to understand without further clarification. For example, the relationship between the MA-KID curve and the model's sensitivity is not entirely clear from the current explanation. 

2. The proposed fast sensitivity analysis is the key component of the paper, but its description is rather technical and lacks an intuitive explanation of why it works and how it significantly reduces the computational cost. 

3. The iterative estimation process and the use of Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) could be better illustrated with examples or visualizations. 

4. In comparison with other data augmentation techniques, the paper could explore more about the reasons why some methods perform better or worse in different scenarios. For instance, a deeper analysis of why TrivialAugment has certain limitations compared to the proposed method could provide more insights into the novelty and superiority of the proposed approach.

5. Fig. 1 is confusion. It looks more like a result of the analysis and does not present the complete pipeline of the method. I suggest redrawing Fig. 1 so that this figure can reflect the steps of the method (Overview of the proposed method).

### Questions
See "Weaknesses"

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The manuscript proposes a novel sensitivity-based data augmentation method to enhance the robustness of semantic segmentation models against natural corruptions. This method significantly improves upon existing approaches in terms of speed and storage efficiency, with runtimes up to 10 times faster and storage requirements reduced by 200 times. It dynamically adjusts augmentations based on the model's sensitivity, thereby enhancing robustness across various real-world and synthetic datasets.

### Strengths
1. The paper proposes an efficient sensitivity analysis method, which is 10 times faster and reduces storage requirements by 200 times compared to existing methods. This improvement is particularly useful during online training.
2. Through gradient-free sensitivity analysis and dynamic sampling techniques, this method enables real-time data augmentation during training without the need for additional models or complex computational resources.

### Weaknesses
1. Lack of specific ablation experiments; the performance improvement could be attributed to the pre-trained weights rather than the method proposed in the manuscript. The absence of detailed ablation comparisons may lead to confusion.
2. The comparison is limited to only a few existing data augmentation methods, lacking a comparison with the latest methods from 2024 to comprehensively assess its superiority.
3. As the authors mentioned, they did not fully consider the performance of the method under low-light conditions. However, in complex weather like snow, low-light conditions often accompany, making it challenging to assess its contribution to the community.
4. The paper treats all types of augmentations as equivalent, but is the contribution of all data augmentations the same in practical environmental conditions? The authors might need to attempt visualizations to specifically present this.
5. The explanation of hyperparameters is not detailed, including but not limited to the settings of hyperparameters on the original model (although this could be elaborated in the appendix, it seems the authors overlooked this). Also, are the same hyperparameters used across different datasets and environmental conditions?
6. There are some minor writing issues, such as most formulas lacking punctuation, and it is suggested that the last page of the manuscript should not be left blank.

### Questions
1. Can the method maintain robustness when dealing with other types of disturbances beyond natural ones, such as artificial image editing?
2. What are the computational resource requirements for the method mentioned in the manuscript on datasets of different scales? Is it possible to effectively deploy it in resource-constrained environments?
3. What challenges does the proposed method face in handling physical occlusions (such as windshield wipers)? Can physical occlusions be simultaneously treated as a complex environment to address?
4. What are the computational resource requirements for the method mentioned in the paper on datasets of different scales? Is it possible to effectively deploy it in resource-constrained driving environments?

### Soundness
3

### Presentation
2

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
This work aims to optimize data augmentation policies to enhance semantic segmentation, particularly targeting domain-shift issues. The proposed approach utilizes a derivative-based local method (first-order indices), approximated with Monte Carlo, to derive importance weights for each augmentation policy. Subsequently, all policies are sampled based on their importance and ensembled to create an augmented data pool. The resulting performance gains are validated through experiments across various domain-shift datasets, demonstrating the effectiveness of the method.

### Strengths
1. The experiments demonstrate that augmentation with policy optimization effectively enhances semantic segmentation modeling.
2. The proposed approach is presented in a clear and organized manner, making it easy to follow. The appendix provides valuable details on the augmentation configuration, enhancing understanding.

### Weaknesses
1. The contributions and insights of this work are somewhat limited. It primarily focuses on enhancing performance through augmentation policy optimization. The sensitivity assessment resembles model uncertainty computation, and uncertainty-weighted segmentation modeling has already been developed to improve model performance. Additionally, this work may raise concerns about relying on performance-boosting tricks rather than providing critical scientific research insights.
2. The experiments lack alignment in terms of using the same semantic segmentation models. The varying choices, without clear justification, could raise concerns about the potential cherry-picking of results. The chosen baseline models, particularly for clean image evaluation, are relatively weak and may not adequately reflect state-of-the-art performance.
3. The augmentation benchmarking experiments are not particularly convincing. The chosen baselines focus on the design of augmentation algorithms, while the proposed method resembles an ensemble of multiple augmentation algorithms, optimizing policy ensembling with importance weighting. Also, the policy optimization is quite common in machine learning model training. It would be beneficial to compare this approach with other policy-based optimization methods.

### Questions
The weaknesses mentioned above raise several questions. The rebuttal should address these concerns to improve the rating.

### Soundness
3

### Presentation
3

### Contribution
2
